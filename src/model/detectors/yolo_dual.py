from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.backbones.dual_backbone import AsymmetricDualBackbone
from src.model.heads.obb_decoupled_head import OBBDecoupledHead
from src.model.necks.enhanced_neck import EnhancedNeck
from src.model.temporal.temporal_fpn import TemporalFeaturePyramid
from src.model.temporal.temporal_memory import TemporalMemoryFusion
from src.registry.fusion_registry import FUSIONS
from src.registry.model_registry import DETECTORS


@DETECTORS.register("YOLODualModalOBB")
class YOLODualModalOBB(nn.Module):
    def __init__(self, num_classes=5, channels=[64, 128, 256, 512], norm_type='GN',
                 use_contrastive=False, fusion_att_type='DualStreamFusion',
                 temporal_enabled=False, temporal_stride=1, fusion=None, temporal=None):
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
            temporal_mode = 'two_frame' if temporal_enabled else 'off'
        if temporal_mode not in {'off', 'two_frame', 'memory'}:
            raise ValueError(f"Unsupported temporal mode: {temporal_mode}")

        self.temporal_enabled = temporal_mode != 'off'
        self.temporal_mode = temporal_mode
        self.temporal_memory_len = temporal_cfg.get('memory_len', 3)
        self.temporal_aggregator = temporal_cfg.get('aggregator', 'weighted_avg')
        self.temporal_gate_hidden_ratio = temporal_cfg.get('gate_hidden_ratio', 0.25)

        fusion_aliases = {
            'JCA': 'DualStreamFusion',
            'CBAM': 'DualStreamFusion',
            'RDM': 'RDMFusion',
        }
        self.backbone = AsymmetricDualBackbone(channels=self.channels, norm_type=self.norm_type)
        if fusion is not None and 'type' in fusion:
            fusion_cfg = deepcopy(dict(fusion))
            fusion_cfg.setdefault('channel_list', self.channels)
            self.fusion = FUSIONS.build(fusion_cfg)
        else:
            fusion_type = fusion_aliases.get(self.fusion_att_type, self.fusion_att_type)
            self.fusion = FUSIONS.build({'type': fusion_type, 'channel_list': self.channels})
        self.neck = EnhancedNeck(channels=self.channels)
        self.temporal_fpn = TemporalFeaturePyramid(channels=self.channels) if self.temporal_mode == 'two_frame' else None
        self.temporal_memory_fusion = (
            TemporalMemoryFusion(
                channels=self.channels,
                memory_len=self.temporal_memory_len,
                aggregator=self.temporal_aggregator,
                gate_hidden_ratio=self.temporal_gate_hidden_ratio,
            )
            if self.temporal_mode == 'memory' else None
        )
        self.head = OBBDecoupledHead(num_classes=self.nc, channels=self.channels, return_dict=True)

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

    def reset_temporal_memory(self):
        self.temporal_memory = []

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
            prev_enhanced_feats, _, _, _, _ = self._extract_features(prev_rgb, prev_ir, return_attention_map=False)
            return [prev_enhanced_feats]
        return []

    def forward(self, img_rgb, img_ir, prev_rgb=None, prev_ir=None, memory_feats=None, return_attention_map=False):
        self.last_temporal_state = None
        enhanced_feats, feat_rgb, feat_ir, att_maps, target_hw = self._extract_features(
            img_rgb, img_ir, return_attention_map=return_attention_map
        )

        temporal_maps = {}
        if self.temporal_mode == 'two_frame':
            current_feats_before_temporal = enhanced_feats
            if prev_rgb is None or prev_ir is None:
                prev_enhanced_feats = self._clone_feature_tuple(enhanced_feats)
            else:
                prev_enhanced_feats, _, _, _, _ = self._extract_features(prev_rgb, prev_ir, return_attention_map=False)

            enhanced_feats, temporal_maps = self.temporal_fpn(
                enhanced_feats,
                prev_enhanced_feats,
                return_attention_map=True,
                target_size=target_hw if return_attention_map else None
            )
            self.last_temporal_state = {
                'current_feats_before_temporal': current_feats_before_temporal,
                'current_feats_after_temporal': enhanced_feats,
                'reference_feats': prev_enhanced_feats,
                'temporal_maps': temporal_maps,
            }
        elif self.temporal_mode == 'memory':
            current_feats_before_temporal = enhanced_feats
            resolved_memory_steps = self._resolve_memory_steps(memory_feats=memory_feats, prev_rgb=prev_rgb, prev_ir=prev_ir)
            enhanced_feats, temporal_maps = self.temporal_memory_fusion(
                enhanced_feats,
                resolved_memory_steps,
                return_attention_map=True,
                target_size=target_hw if return_attention_map else None
            )
            reference_feats = resolved_memory_steps[-1] if resolved_memory_steps else self._clone_feature_tuple(current_feats_before_temporal)
            self.last_temporal_state = {
                'current_feats_before_temporal': current_feats_before_temporal,
                'current_feats_after_temporal': enhanced_feats,
                'reference_feats': reference_feats,
                'temporal_maps': temporal_maps,
            }
            self.update_temporal_memory(current_feats_before_temporal)

        outputs = self.head(enhanced_feats)

        if return_attention_map:
            merged_maps = {}
            if att_maps:
                merged_maps.update(att_maps)
            if temporal_maps:
                merged_maps.update(temporal_maps)
            return outputs, merged_maps, feat_rgb, feat_ir
        return outputs, feat_rgb, feat_ir

    def get_temporal_consistency_loss(self, lambda_t=0.1, low_motion_bias=0.75):
        if not self.temporal_enabled or self.last_temporal_state is None:
            return None

        current_feats = self.last_temporal_state['current_feats_after_temporal']
        reference_feats = self.last_temporal_state['reference_feats']
        temporal_maps = self.last_temporal_state['temporal_maps']

        losses = []
        level_names = ['P2', 'P3', 'P4', 'P5']
        for level_name, current_feat, prev_feat in zip(level_names, current_feats, reference_feats):
            motion_map = temporal_maps.get(f'{level_name}_Temporal_Map')
            if motion_map is None or motion_map.shape[-2:] != current_feat.shape[-2:]:
                motion_map = torch.ones_like(current_feat[:, :1])

            low_motion_weight = (1.0 - motion_map.detach()).clamp(min=0.0, max=1.0)
            low_motion_weight = low_motion_bias + (1.0 - low_motion_bias) * low_motion_weight

            pooled_current = F.adaptive_avg_pool2d(current_feat * low_motion_weight, 1).flatten(1)
            pooled_prev = F.adaptive_avg_pool2d(prev_feat.detach() * low_motion_weight, 1).flatten(1)
            pooled_current = F.normalize(pooled_current, dim=1)
            pooled_prev = F.normalize(pooled_prev, dim=1)
            cosine_term = 1.0 - (pooled_current * pooled_prev).sum(dim=1)

            spatial_term = F.smooth_l1_loss(
                current_feat * low_motion_weight,
                prev_feat.detach() * low_motion_weight,
                reduction='none'
            ).mean(dim=(1, 2, 3))
            losses.append((cosine_term + spatial_term).mean())

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
