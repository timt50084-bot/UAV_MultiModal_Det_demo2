from copy import deepcopy
import inspect
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.backbones.dual_backbone import AsymmetricDualBackbone
from src.model.heads.obb_decoupled_head import OBBDecoupledHead
from src.model.necks.enhanced_neck import EnhancedNeck
from src.model.temporal.temporal_fpn import TemporalFeaturePyramid
from src.registry.fusion_registry import FUSIONS
from src.registry.model_registry import DETECTORS


LEGACY_FUSION_ALIASES = {
    'JCA': 'DualStreamFusion',
    'CBAM': 'DualStreamFusion',
    'RDM': 'RDMFusion',
}


@DETECTORS.register("YOLODualModalOBB")
class YOLODualModalOBB(nn.Module):
    def __init__(
        self,
        num_classes=5,
        channels=[64, 128, 256, 512],
        norm_type='GN',
        use_contrastive=False,
        fusion_att_type=None,
        temporal_enabled=False,
        temporal_stride=1,
        fusion=None,
        temporal=None,
    ):
        super().__init__()
        self.nc = num_classes
        self.channels = channels
        self.use_contrastive = use_contrastive
        self.norm_type = norm_type
        self.fusion_att_type = fusion_att_type
        self.temporal_stride = temporal_stride
        self.last_temporal_state = None
        self.temporal_memory = []

        temporal_cfg = deepcopy(dict(temporal)) if temporal is not None else {}
        temporal_enabled = temporal_cfg.get('enabled', temporal_enabled)
        temporal_mode = temporal_cfg.get('mode')
        if temporal_mode is None:
            # The maintained detector mainline defaults to two-frame temporal
            # whenever temporal is enabled. The memory route remains a legacy
            # compatibility path for archived detection experiments only.
            temporal_mode = 'two_frame' if temporal_enabled else 'off'
        if temporal_mode not in {'off', 'two_frame', 'memory'}:
            raise ValueError(f"Unsupported temporal mode: {temporal_mode}")

        self.temporal_enabled = temporal_mode != 'off'
        self.temporal_mode = temporal_mode
        self.temporal_memory_len = temporal_cfg.get('memory_len', 3)
        self.temporal_aggregator = temporal_cfg.get('aggregator', 'weighted_avg')
        self.temporal_gate_hidden_ratio = temporal_cfg.get('gate_hidden_ratio', 0.25)
        self.temporal_consistency_feature_clip = float(temporal_cfg.get('consistency_feature_clip', 5.0))
        self.temporal_consistency_level_cap = float(temporal_cfg.get('consistency_level_cap', 2.0))
        self.temporal_consistency_eps = float(temporal_cfg.get('consistency_eps', 1e-6))

        self.backbone = AsymmetricDualBackbone(channels=self.channels, norm_type=self.norm_type)
        self.fusion = self._build_fusion_module(fusion, fusion_att_type)
        self.neck = EnhancedNeck(channels=self.channels)
        self.temporal_fpn = TemporalFeaturePyramid(channels=self.channels) if self.temporal_mode == 'two_frame' else None
        self.temporal_memory_fusion = self._build_temporal_memory_fusion() if self.temporal_mode == 'memory' else None
        self.head = OBBDecoupledHead(num_classes=self.nc, channels=self.channels, return_dict=True)

    def _build_fusion_module(self, fusion, fusion_att_type):
        if fusion is not None and fusion.get('type'):
            fusion_cfg = deepcopy(dict(fusion))
            fusion_cfg.setdefault('channel_list', self.channels)
            fusion_cfg = self._filter_unsupported_fusion_args(fusion_cfg)
            return FUSIONS.build(fusion_cfg)

        # Keep the old fusion_att_type route as a thin compatibility layer only.
        fusion_type = LEGACY_FUSION_ALIASES.get(fusion_att_type, fusion_att_type)
        if fusion_type is None:
            fusion_type = 'DualStreamFusion'
        else:
            warnings.warn(
                'model.fusion_att_type is deprecated; prefer model.fusion.type in configs and direct model builds.',
                DeprecationWarning,
                stacklevel=3,
            )

        return FUSIONS.build({'type': fusion_type, 'channel_list': self.channels})

    def _filter_unsupported_fusion_args(self, fusion_cfg):
        fusion_type = fusion_cfg.get('type')
        fusion_cls = FUSIONS._module_dict.get(fusion_type)
        if fusion_cls is None:
            return fusion_cfg

        signature = inspect.signature(fusion_cls.__init__)
        if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values()):
            return fusion_cfg

        allowed_keys = {
            name
            for name, parameter in signature.parameters.items()
            if name != 'self' and parameter.kind in {
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            }
        }
        filtered_cfg = {'type': fusion_type}
        dropped_keys = []
        for key, value in fusion_cfg.items():
            if key == 'type':
                continue
            if key in allowed_keys:
                filtered_cfg[key] = value
            else:
                dropped_keys.append(key)

        if dropped_keys:
            warnings.warn(
                f"Dropping unsupported fusion args for {fusion_type}: {', '.join(sorted(dropped_keys))}.",
                stacklevel=3,
            )
        return filtered_cfg

    def _build_temporal_memory_fusion(self):
        # Detection-side temporal memory is no longer the maintained mainline.
        # Lazy import keeps historical experiments working without making it
        # look like the default detector path in builder.py.
        warnings.warn(
            "Detection temporal mode 'memory' is deprecated; the maintained mainline uses model.temporal.mode='two_frame'.",
            DeprecationWarning,
            stacklevel=3,
        )
        from src.model.temporal.temporal_memory import TemporalMemoryFusion

        return TemporalMemoryFusion(
            channels=self.channels,
            memory_len=self.temporal_memory_len,
            aggregator=self.temporal_aggregator,
            gate_hidden_ratio=self.temporal_gate_hidden_ratio,
        )

    def _extract_features(self, img_rgb, img_ir, return_attention_map=False, target_size=None):
        h, w = img_rgb.shape[2:]

        feat_rgb, feat_ir = self.backbone(img_rgb, img_ir)

        if return_attention_map:
            fused_feats, att_maps = self.fusion(feat_rgb, feat_ir, return_attention_map=True, target_size=(h, w))
        else:
            fused_feats = self.fusion(feat_rgb, feat_ir)

        enhanced_feats = self.neck(fused_feats)

        if return_attention_map:
            return enhanced_feats, feat_rgb, feat_ir, att_maps, (h, w)
        return enhanced_feats, feat_rgb, feat_ir, None, (h, w)

    def _clone_feature_tuple(self, feats):
        return tuple(feat.detach().clone() for feat in feats)

    @staticmethod
    def _detach_feature_tuple(feats):
        if feats is None:
            return None
        return tuple(feat.detach() for feat in feats)

    @staticmethod
    def _detach_temporal_maps(temporal_maps):
        if not temporal_maps:
            return {}
        return {name: value.detach() for name, value in temporal_maps.items()}

    def _extract_reference_features(self, img_rgb, img_ir):
        with torch.no_grad():
            enhanced_feats, _, _, _, _ = self._extract_features(
                img_rgb,
                img_ir,
                return_attention_map=False,
            )
        return self._detach_feature_tuple(enhanced_feats)

    def _build_temporal_state(
        self,
        current_feats_before_temporal,
        current_feats_after_temporal,
        reference_feats,
        temporal_maps,
        reference_valid,
    ):
        return {
            'current_feats_before_temporal': self._detach_feature_tuple(current_feats_before_temporal),
            'current_feats_after_temporal': current_feats_after_temporal if reference_valid else None,
            'reference_feats': self._detach_feature_tuple(reference_feats),
            'temporal_maps': self._detach_temporal_maps(temporal_maps),
            'reference_valid': bool(reference_valid),
        }

    def _build_tracking_feature_payload(self, enhanced_feats, feat_rgb, feat_ir, input_hw):
        return {
            'fused_feats': tuple(feat.detach() for feat in enhanced_feats),
            'rgb_feats': tuple(feat.detach() for feat in feat_rgb),
            'ir_feats': tuple(feat.detach() for feat in feat_ir),
            'input_hw': tuple(int(value) for value in input_hw),
        }

    def reset_temporal_memory(self):
        self.reset_temporal_state(clear_memory=True)

    def clear_temporal_step_state(self):
        self.last_temporal_state = None

    def reset_temporal_state(self, clear_memory=True):
        self.clear_temporal_step_state()
        if clear_memory:
            self.temporal_memory = []

    def get_temporal_debug_state(self):
        return {
            'enabled': bool(self.temporal_enabled),
            'mode': self.temporal_mode,
            'memory_size': len(self.temporal_memory),
            'memory_len': int(self.temporal_memory_len),
            'has_step_state': self.last_temporal_state is not None,
        }

    def get_temporal_memory(self):
        return tuple(self.temporal_memory)

    def update_temporal_memory(self, feats):
        if feats is None:
            return
        self.temporal_memory.append(self._clone_feature_tuple(feats))
        if len(self.temporal_memory) > self.temporal_memory_len:
            self.temporal_memory = self.temporal_memory[-self.temporal_memory_len:]

    def _resolve_memory_steps(self, memory_feats=None, prev_rgb=None, prev_ir=None):
        if memory_feats is not None:
            if isinstance(memory_feats, (list, tuple)) and len(memory_feats) > 0 and torch.is_tensor(memory_feats[0]):
                return [tuple(memory_feats)]
            return list(memory_feats)
        if self.temporal_memory:
            return list(self.temporal_memory)
        if prev_rgb is not None and prev_ir is not None:
            return [self._extract_reference_features(prev_rgb, prev_ir)]
        return []

    def forward(
        self,
        img_rgb,
        img_ir,
        prev_rgb=None,
        prev_ir=None,
        memory_feats=None,
        return_attention_map=False,
        return_tracking_features=False,
    ):
        self.last_temporal_state = None
        enhanced_feats, feat_rgb, feat_ir, att_maps, target_hw = self._extract_features(
            img_rgb,
            img_ir,
            return_attention_map=return_attention_map,
        )

        temporal_maps = {}
        if self.temporal_mode == 'two_frame':
            current_feats_before_temporal = enhanced_feats
            reference_valid = prev_rgb is not None and prev_ir is not None
            prev_enhanced_feats = None
            if reference_valid:
                prev_enhanced_feats = self._extract_reference_features(prev_rgb, prev_ir)
                enhanced_feats, temporal_maps = self.temporal_fpn(
                    enhanced_feats,
                    prev_enhanced_feats,
                    return_attention_map=True,
                    target_size=target_hw if return_attention_map else None,
                )
            self.last_temporal_state = self._build_temporal_state(
                current_feats_before_temporal=current_feats_before_temporal,
                current_feats_after_temporal=enhanced_feats,
                reference_feats=prev_enhanced_feats,
                temporal_maps=temporal_maps,
                reference_valid=reference_valid,
            )
        elif self.temporal_mode == 'memory':
            # Compatibility-only branch for archived detection experiments.
            current_feats_before_temporal = enhanced_feats
            resolved_memory_steps = self._resolve_memory_steps(
                memory_feats=memory_feats,
                prev_rgb=prev_rgb,
                prev_ir=prev_ir,
            )
            enhanced_feats, temporal_maps = self.temporal_memory_fusion(
                enhanced_feats,
                resolved_memory_steps,
                return_attention_map=True,
                target_size=target_hw if return_attention_map else None,
            )
            reference_valid = bool(resolved_memory_steps)
            reference_feats = resolved_memory_steps[-1] if resolved_memory_steps else None
            self.last_temporal_state = self._build_temporal_state(
                current_feats_before_temporal=current_feats_before_temporal,
                current_feats_after_temporal=enhanced_feats,
                reference_feats=reference_feats,
                temporal_maps=temporal_maps,
                reference_valid=reference_valid,
            )
            self.update_temporal_memory(current_feats_before_temporal)

        outputs = self.head(enhanced_feats)
        tracking_payload = (
            self._build_tracking_feature_payload(enhanced_feats, feat_rgb, feat_ir, target_hw)
            if return_tracking_features else None
        )

        if return_attention_map:
            merged_maps = {}
            if att_maps:
                merged_maps.update(att_maps)
            if temporal_maps:
                merged_maps.update(temporal_maps)
            if return_tracking_features:
                return outputs, merged_maps, feat_rgb, feat_ir, tracking_payload
            return outputs, merged_maps, feat_rgb, feat_ir
        if return_tracking_features:
            return outputs, feat_rgb, feat_ir, tracking_payload
        return outputs, feat_rgb, feat_ir

    def get_temporal_consistency_loss(self, lambda_t=0.1, low_motion_bias=0.75):
        if not self.temporal_enabled or self.last_temporal_state is None:
            return None

        current_feats = self.last_temporal_state['current_feats_after_temporal']
        reference_feats = self.last_temporal_state['reference_feats']
        temporal_maps = self.last_temporal_state['temporal_maps']
        if current_feats is None or not self.last_temporal_state.get('reference_valid', False) or not reference_feats:
            return None

        losses = []
        level_names = ['P2', 'P3', 'P4', 'P5']
        for level_name, current_feat, prev_feat in zip(level_names, current_feats, reference_feats):
            motion_map = temporal_maps.get(f'{level_name}_Temporal_Map')
            if motion_map is None or motion_map.shape[-2:] != current_feat.shape[-2:]:
                motion_map = torch.ones_like(current_feat[:, :1])

            low_motion_weight = (1.0 - motion_map.detach()).clamp(min=0.0, max=1.0)
            low_motion_weight = low_motion_bias + (1.0 - low_motion_bias) * low_motion_weight

            current_feat = torch.nan_to_num(
                current_feat,
                nan=0.0,
                posinf=self.temporal_consistency_feature_clip,
                neginf=-self.temporal_consistency_feature_clip,
            )
            prev_feat = torch.nan_to_num(
                prev_feat.detach(),
                nan=0.0,
                posinf=self.temporal_consistency_feature_clip,
                neginf=-self.temporal_consistency_feature_clip,
            )
            feature_scale = torch.maximum(
                current_feat.detach().abs().mean(dim=(1, 2, 3), keepdim=True),
                prev_feat.abs().mean(dim=(1, 2, 3), keepdim=True),
            ).clamp(min=1.0)
            current_weighted = ((current_feat / feature_scale) * low_motion_weight).clamp(
                min=-self.temporal_consistency_feature_clip,
                max=self.temporal_consistency_feature_clip,
            )
            prev_weighted = ((prev_feat / feature_scale) * low_motion_weight).clamp(
                min=-self.temporal_consistency_feature_clip,
                max=self.temporal_consistency_feature_clip,
            )

            pooled_current = F.adaptive_avg_pool2d(current_weighted, 1).flatten(1)
            pooled_prev = F.adaptive_avg_pool2d(prev_weighted, 1).flatten(1)
            pooled_current = F.normalize(pooled_current, dim=1, eps=self.temporal_consistency_eps)
            pooled_prev = F.normalize(pooled_prev, dim=1, eps=self.temporal_consistency_eps)
            cosine_term = (1.0 - (pooled_current * pooled_prev).sum(dim=1)).clamp(min=0.0, max=2.0)

            spatial_term = F.smooth_l1_loss(
                current_weighted,
                prev_weighted,
                reduction='none',
            ).mean(dim=(1, 2, 3)).clamp(min=0.0, max=self.temporal_consistency_level_cap)
            level_loss = torch.nan_to_num(
                cosine_term + spatial_term,
                nan=0.0,
                posinf=self.temporal_consistency_level_cap,
                neginf=0.0,
            ).clamp(min=0.0, max=self.temporal_consistency_level_cap)
            losses.append(level_loss.mean())

        return current_feats[0].new_tensor(lambda_t) * torch.stack(losses).mean()

    def get_contrastive_alignment_loss(self, rgb_feats, ir_feats, temperature=0.07, lambda_c=0.1):
        if not self.use_contrastive:
            return torch.tensor(0.0, device=rgb_feats[0].device)

        r_vec = F.normalize(torch.mean(rgb_feats[-1], dim=[2, 3]), dim=1)
        i_vec = F.normalize(torch.mean(ir_feats[-1], dim=[2, 3]), dim=1)

        logits = torch.matmul(r_vec, i_vec.t()) / temperature
        labels = torch.arange(r_vec.size(0), device=r_vec.device)

        loss_r2i = F.cross_entropy(logits, labels)
        loss_i2r = F.cross_entropy(logits.t(), labels)

        return (loss_r2i + loss_i2r) * 0.5 * lambda_c
