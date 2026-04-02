import warnings
from collections.abc import Mapping

from src.engine.evaluator import Evaluator, GPUDetectionEvaluator
from src.metrics.obb_iou_backend import (
    OBB_IOU_BACKEND_CPU_POLYGON,
    OBB_IOU_BACKEND_GPU_PROB,
    resolve_obb_iou_backend_name,
)
from src.metrics.obb_metrics import GPUOBBMetricsEvaluator, OBBMetricsEvaluator
from src.metrics.task_metrics import normalize_eval_metrics_cfg

try:
    from omegaconf import OmegaConf
except ImportError:  # pragma: no cover
    OmegaConf = None


DETECTION_EVALUATOR_CPU = 'cpu'
DETECTION_EVALUATOR_GPU = 'gpu'


def _config_to_dict(cfg):
    if cfg is None:
        return {}
    if OmegaConf is not None and OmegaConf.is_config(cfg):
        return OmegaConf.to_container(cfg, resolve=True)
    if isinstance(cfg, Mapping):
        return dict(cfg)
    if hasattr(cfg, 'items'):
        return dict(cfg.items())
    return {}


def get_detection_evaluator_backend(eval_cfg=None):
    eval_cfg = _config_to_dict(eval_cfg)
    backend = str(eval_cfg.get('evaluator', DETECTION_EVALUATOR_GPU)).strip().lower()
    return backend or DETECTION_EVALUATOR_GPU


def _resolve_requested_obb_iou_backend(eval_cfg, evaluator_backend):
    if 'obb_iou_backend' in eval_cfg and eval_cfg.get('obb_iou_backend') is not None:
        return resolve_obb_iou_backend_name(eval_cfg)
    if evaluator_backend == DETECTION_EVALUATOR_GPU:
        return OBB_IOU_BACKEND_GPU_PROB
    return OBB_IOU_BACKEND_CPU_POLYGON


def _build_cpu_eval_cfg(eval_cfg, obb_iou_backend=OBB_IOU_BACKEND_CPU_POLYGON):
    resolved_cfg = dict(eval_cfg)
    resolved_cfg['evaluator'] = DETECTION_EVALUATOR_CPU
    resolved_cfg['obb_iou_backend'] = obb_iou_backend
    return resolved_cfg


def _annotate_evaluator_resolution(
    evaluator,
    requested_backend,
    requested_obb_iou_backend,
    resolved_backend,
    resolved_obb_iou_backend,
    evaluator_role,
    resolution_reason,
):
    evaluator.requested_backend = requested_backend
    evaluator.requested_obb_iou_backend = requested_obb_iou_backend
    evaluator.resolved_backend = resolved_backend
    evaluator.resolved_obb_iou_backend = resolved_obb_iou_backend
    evaluator.evaluator_role = evaluator_role
    evaluator.resolution_reason = resolution_reason
    return evaluator


def _warn_gpu_evaluator_limits(eval_cfg):
    normalized = normalize_eval_metrics_cfg(eval_cfg)
    notes = []
    if normalized['small_object'].get('enabled', True):
        notes.append('GPU evaluator emits mAP_S only; Recall_S and Precision_S remain unavailable')
    if normalized['temporal_stability'].get('enabled', 'auto') in (True, 'auto'):
        notes.append('TemporalStability remains outside the GPU evaluator path')
    if normalized['group_eval'].get('enabled', False):
        notes.append('GroupedMetrics remain outside the GPU evaluator path')
    if normalized['error_analysis'].get('enabled', False):
        notes.append('ErrorAnalysis still falls back to the CPU reference analysis path')
    if notes:
        warnings.warn('GPU detection evaluator coverage: ' + '; '.join(notes) + '.', UserWarning)


def _warn_gpu_fallback_to_cpu(device):
    warnings.warn(
        "Detection evaluator backend 'gpu' was requested but the active device "
        f"is '{device}', which is not CUDA. Falling back to the CPU reference "
        "evaluator with eval.evaluator='cpu' and eval.obb_iou_backend='cpu_polygon'. "
        "Set those values explicitly if you want to pin the reference path.",
        UserWarning,
        stacklevel=2,
    )


def _warn_cpu_reference_backend_override(requested_obb_iou_backend):
    warnings.warn(
        "Detection evaluator backend 'cpu' is now a reference/fallback path and "
        "always resolves to eval.obb_iou_backend='cpu_polygon'. "
        f"Ignoring requested eval.obb_iou_backend='{requested_obb_iou_backend}'.",
        UserWarning,
        stacklevel=2,
    )


def build_detection_evaluator(dataloader, device, num_classes, nms_kwargs=None, eval_cfg=None, infer_cfg=None):
    eval_cfg = _config_to_dict(eval_cfg)
    requested_backend = get_detection_evaluator_backend(eval_cfg)
    requested_obb_iou_backend = _resolve_requested_obb_iou_backend(eval_cfg, requested_backend)

    if requested_backend == DETECTION_EVALUATOR_CPU:
        resolved_obb_iou_backend = OBB_IOU_BACKEND_CPU_POLYGON
        if requested_obb_iou_backend != resolved_obb_iou_backend:
            _warn_cpu_reference_backend_override(requested_obb_iou_backend)
        resolved_eval_cfg = _build_cpu_eval_cfg(eval_cfg, obb_iou_backend=resolved_obb_iou_backend)
        metrics_evaluator = OBBMetricsEvaluator(
            num_classes=num_classes,
            extra_metrics_cfg=resolved_eval_cfg,
        )
        evaluator = Evaluator(
            dataloader,
            metrics_evaluator,
            device,
            nms_kwargs=nms_kwargs,
            extra_metrics_cfg=resolved_eval_cfg,
            infer_cfg=infer_cfg,
        )
        return _annotate_evaluator_resolution(
            evaluator,
            requested_backend=requested_backend,
            requested_obb_iou_backend=requested_obb_iou_backend,
            resolved_backend=DETECTION_EVALUATOR_CPU,
            resolved_obb_iou_backend=resolved_obb_iou_backend,
            evaluator_role='reference',
            resolution_reason='explicit_cpu_reference',
        )

    if requested_backend == DETECTION_EVALUATOR_GPU:
        if requested_obb_iou_backend != OBB_IOU_BACKEND_GPU_PROB:
            raise ValueError(
                "Detection evaluator backend 'gpu' requires eval.obb_iou_backend='gpu_prob' "
                f"but received '{requested_obb_iou_backend}'."
            )
        resolved_eval_cfg = dict(eval_cfg)
        resolved_eval_cfg['evaluator'] = DETECTION_EVALUATOR_GPU
        resolved_eval_cfg['obb_iou_backend'] = OBB_IOU_BACKEND_GPU_PROB
        if str(getattr(device, 'type', device)) != 'cuda':
            _warn_gpu_fallback_to_cpu(device)
            resolved_eval_cfg = _build_cpu_eval_cfg(eval_cfg, obb_iou_backend=OBB_IOU_BACKEND_CPU_POLYGON)
            metrics_evaluator = OBBMetricsEvaluator(
                num_classes=num_classes,
                extra_metrics_cfg=resolved_eval_cfg,
            )
            evaluator = Evaluator(
                dataloader,
                metrics_evaluator,
                device,
                nms_kwargs=nms_kwargs,
                extra_metrics_cfg=resolved_eval_cfg,
                infer_cfg=infer_cfg,
            )
            return _annotate_evaluator_resolution(
                evaluator,
                requested_backend=requested_backend,
                requested_obb_iou_backend=requested_obb_iou_backend,
                resolved_backend=DETECTION_EVALUATOR_CPU,
                resolved_obb_iou_backend=OBB_IOU_BACKEND_CPU_POLYGON,
                evaluator_role='fallback',
                resolution_reason='non_cuda_fallback',
            )
        _warn_gpu_evaluator_limits(resolved_eval_cfg)
        metrics_evaluator = GPUOBBMetricsEvaluator(
            num_classes=num_classes,
            device=device,
            extra_metrics_cfg=resolved_eval_cfg,
        )
        evaluator = GPUDetectionEvaluator(
            dataloader,
            metrics_evaluator,
            device,
            nms_kwargs=nms_kwargs,
            extra_metrics_cfg=resolved_eval_cfg,
            infer_cfg=infer_cfg,
        )
        return _annotate_evaluator_resolution(
            evaluator,
            requested_backend=requested_backend,
            requested_obb_iou_backend=requested_obb_iou_backend,
            resolved_backend=DETECTION_EVALUATOR_GPU,
            resolved_obb_iou_backend=OBB_IOU_BACKEND_GPU_PROB,
            evaluator_role='mainline',
            resolution_reason='gpu_mainline',
        )

    raise NotImplementedError(
        f"Detection evaluator backend '{requested_backend}' is not implemented. "
        f"Supported backends: '{DETECTION_EVALUATOR_CPU}', '{DETECTION_EVALUATOR_GPU}'."
    )
