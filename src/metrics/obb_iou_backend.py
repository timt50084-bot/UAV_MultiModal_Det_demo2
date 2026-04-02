from collections.abc import Mapping

import numpy as np

try:
    from shapely.geometry import Polygon
except ImportError:  # pragma: no cover - optional in some lightweight test envs
    Polygon = None

try:
    import torch
    from src.model.bbox_utils import batch_prob_iou
except ImportError:  # pragma: no cover - optional in config-only test environments
    torch = None
    batch_prob_iou = None

from src.utils.detection_cuda import DETECTION_CUDA_REQUIRED_MESSAGE


OBB_IOU_BACKEND_GPU_PROB = 'gpu_prob'

_OBB_IOU_BACKEND_ALIASES = {
    'gpu': OBB_IOU_BACKEND_GPU_PROB,
    'prob': OBB_IOU_BACKEND_GPU_PROB,
    'prob_iou': OBB_IOU_BACKEND_GPU_PROB,
    'gpu_prob': OBB_IOU_BACKEND_GPU_PROB,
}


def normalize_obb_iou_backend_name(name):
    backend = str(name or OBB_IOU_BACKEND_GPU_PROB).strip().lower()
    return _OBB_IOU_BACKEND_ALIASES.get(backend, backend or OBB_IOU_BACKEND_GPU_PROB)


def resolve_obb_iou_backend_name(cfg=None):
    if isinstance(cfg, Mapping):
        return normalize_obb_iou_backend_name(cfg.get('obb_iou_backend', OBB_IOU_BACKEND_GPU_PROB))
    return normalize_obb_iou_backend_name(cfg)


def obb2polygon(obb):
    if Polygon is None:
        raise RuntimeError(
            'Exact polygon analysis requires shapely, but shapely is not available in the current environment.'
        )
    cx, cy, w, h, theta = obb
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    dx, dy = w / 2.0, h / 2.0
    pts = np.array([[-dx, -dy], [dx, -dy], [dx, dy], [-dx, dy]], dtype=np.float32)
    rotation = np.array([[cos_t, -sin_t], [sin_t, cos_t]], dtype=np.float32)
    pts = pts @ rotation.T + np.array([cx, cy], dtype=np.float32)
    try:
        poly = Polygon(pts)
        if not poly.is_valid:
            poly = poly.buffer(0)
        return poly
    except Exception:
        return None


def hbb_prescreening(obb1, obb2):
    cx1, cy1, w1, h1, t1 = obb1
    cx2, cy2, w2, h2, t2 = obb2
    w1h = w1 * abs(np.cos(t1)) + h1 * abs(np.sin(t1))
    h1h = w1 * abs(np.sin(t1)) + h1 * abs(np.cos(t1))
    w2h = w2 * abs(np.cos(t2)) + h2 * abs(np.sin(t2))
    h2h = w2 * abs(np.sin(t2)) + h2 * abs(np.cos(t2))
    if abs(cx1 - cx2) > (w1h + w2h) / 2:
        return False
    if abs(cy1 - cy2) > (h1h + h2h) / 2:
        return False
    return True


def polygon_iou(obb1, obb2, poly_cache=None):
    if not hbb_prescreening(obb1, obb2):
        return 0.0

    poly_cache = poly_cache if poly_cache is not None else {}
    key1, key2 = tuple(obb1), tuple(obb2)
    if key1 not in poly_cache:
        poly_cache[key1] = obb2polygon(obb1)
    if key2 not in poly_cache:
        poly_cache[key2] = obb2polygon(obb2)
    poly1, poly2 = poly_cache[key1], poly_cache[key2]
    if poly1 is None or poly2 is None:
        return 0.0
    try:
        inter = poly1.intersection(poly2).area
        union = poly1.area + poly2.area - inter
        return inter / union if union > 0 else 0.0
    except Exception:
        return 0.0


def _normalize_obb_array(boxes):
    if boxes is None:
        return np.zeros((0, 5), dtype=np.float32)

    array = np.asarray(boxes, dtype=np.float32)
    if array.size == 0:
        return np.zeros((0, 5), dtype=np.float32)
    return np.reshape(array, (-1, 5)).astype(np.float32)


def _normalize_obb_tensor(boxes, device):
    if torch is None:
        raise RuntimeError("Torch is required to normalize OBB tensors for the GPU IoU backend.")

    if boxes is None:
        return torch.empty((0, 5), dtype=torch.float32, device=device)
    if isinstance(boxes, torch.Tensor):
        if boxes.numel() == 0:
            return torch.empty((0, 5), dtype=torch.float32, device=device)
        return boxes.to(device=device, dtype=torch.float32).reshape(-1, 5)

    array = _normalize_obb_array(boxes)
    if array.size == 0:
        return torch.empty((0, 5), dtype=torch.float32, device=device)
    return torch.as_tensor(array, dtype=torch.float32, device=device)


class GPUProbIoUBackend:
    name = OBB_IOU_BACKEND_GPU_PROB
    is_exact = False
    supports_tensor_matching = True
    description = 'CUDA-backed ProbIoU surrogate similarity via batch_prob_iou; not exact polygon IoU.'

    def __init__(self, device=None):
        if torch is None or batch_prob_iou is None:
            raise RuntimeError(
                "OBB IoU backend 'gpu_prob' requires torch and the existing batch_prob_iou implementation, "
                'but torch is not available in the current environment.'
            )
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"OBB IoU backend 'gpu_prob' requires CUDA. {DETECTION_CUDA_REQUIRED_MESSAGE}"
            )
        self.device = torch.device(device or 'cuda')

    def pair_iou(self, box_a, box_b):
        return float(self.pairwise_iou_tensor([box_a], [box_b])[0, 0].item())

    def pairwise_iou(self, boxes_a, boxes_b):
        return self.pairwise_iou_tensor(boxes_a, boxes_b).detach().cpu().numpy().astype(np.float32)

    def pairwise_iou_tensor(self, boxes_a, boxes_b):
        tensor_a = _normalize_obb_tensor(boxes_a, self.device)
        tensor_b = _normalize_obb_tensor(boxes_b, self.device)
        if tensor_a.shape[0] == 0 or tensor_b.shape[0] == 0:
            return torch.empty((tensor_a.shape[0], tensor_b.shape[0]), dtype=torch.float32, device=self.device)

        merged = torch.cat([tensor_a, tensor_b], dim=0)
        prob_matrix = batch_prob_iou(merged)
        split = tensor_a.shape[0]
        return prob_matrix[:split, split:]


def build_obb_iou_backend(cfg=None, device=None):
    backend_name = resolve_obb_iou_backend_name(cfg)
    if backend_name == OBB_IOU_BACKEND_GPU_PROB:
        return GPUProbIoUBackend(device=device)
    raise ValueError(
        f"Unsupported OBB IoU backend '{backend_name}'. "
        "Detection now only supports eval.obb_iou_backend='gpu_prob'."
    )
