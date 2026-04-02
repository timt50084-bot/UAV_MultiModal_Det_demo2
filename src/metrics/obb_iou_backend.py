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


OBB_IOU_BACKEND_CPU_POLYGON = 'cpu_polygon'
OBB_IOU_BACKEND_GPU_PROB = 'gpu_prob'

_OBB_IOU_BACKEND_ALIASES = {
    'cpu': OBB_IOU_BACKEND_CPU_POLYGON,
    'polygon': OBB_IOU_BACKEND_CPU_POLYGON,
    'cpu_polygon': OBB_IOU_BACKEND_CPU_POLYGON,
    'gpu': OBB_IOU_BACKEND_GPU_PROB,
    'prob': OBB_IOU_BACKEND_GPU_PROB,
    'prob_iou': OBB_IOU_BACKEND_GPU_PROB,
    'gpu_prob': OBB_IOU_BACKEND_GPU_PROB,
}


def normalize_obb_iou_backend_name(name):
    backend = str(name or OBB_IOU_BACKEND_CPU_POLYGON).strip().lower()
    return _OBB_IOU_BACKEND_ALIASES.get(backend, backend or OBB_IOU_BACKEND_CPU_POLYGON)


def resolve_obb_iou_backend_name(cfg=None):
    if isinstance(cfg, Mapping):
        return normalize_obb_iou_backend_name(cfg.get('obb_iou_backend', OBB_IOU_BACKEND_CPU_POLYGON))
    return normalize_obb_iou_backend_name(cfg)


def obb2polygon(obb):
    if Polygon is None:
        raise RuntimeError(
            "OBB IoU backend 'cpu_polygon' requires shapely, but shapely is not available in the current environment."
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


class CPUPolygonIoUBackend:
    name = OBB_IOU_BACKEND_CPU_POLYGON
    is_exact = True
    supports_tensor_matching = False
    description = 'Exact CPU polygon IoU via shapely intersection/union.'

    def __init__(self):
        self.poly_cache = {}

    def pair_iou(self, box_a, box_b):
        return float(polygon_iou(box_a, box_b, self.poly_cache))

    def pairwise_iou(self, boxes_a, boxes_b):
        boxes_a = _normalize_obb_array(boxes_a)
        boxes_b = _normalize_obb_array(boxes_b)
        if boxes_a.shape[0] == 0 or boxes_b.shape[0] == 0:
            return np.zeros((boxes_a.shape[0], boxes_b.shape[0]), dtype=np.float32)

        output = np.zeros((boxes_a.shape[0], boxes_b.shape[0]), dtype=np.float32)
        for row_idx, box_a in enumerate(boxes_a):
            for col_idx, box_b in enumerate(boxes_b):
                output[row_idx, col_idx] = self.pair_iou(box_a, box_b)
        return output


class GPUProbIoUBackend:
    name = OBB_IOU_BACKEND_GPU_PROB
    is_exact = False
    supports_tensor_matching = True
    description = 'CUDA-backed ProbIoU surrogate similarity via batch_prob_iou; not exact polygon IoU.'

    def __init__(self, device=None):
        if torch is None or batch_prob_iou is None:
            raise RuntimeError(
                "OBB IoU backend 'gpu_prob' requires torch and the existing batch_prob_iou implementation, "
                "but torch is not available in the current environment."
            )
        if not torch.cuda.is_available():
            raise RuntimeError(
                "OBB IoU backend 'gpu_prob' requires CUDA, but torch.cuda.is_available() is False. "
                "Keep eval.obb_iou_backend=cpu_polygon or run on a CUDA-enabled environment."
            )
        self.device = torch.device(device or 'cuda')

    def pair_iou(self, box_a, box_b):
        return float(self.pairwise_iou_tensor([box_a], [box_b])[0, 0].item())

    def pairwise_iou(self, boxes_a, boxes_b):
        return self.pairwise_iou_tensor(boxes_a, boxes_b).detach().cpu().numpy().astype(np.float32)

    def pairwise_iou_tensor(self, boxes_a, boxes_b):
        boxes_a = _normalize_obb_array(boxes_a)
        boxes_b = _normalize_obb_array(boxes_b)
        if boxes_a.shape[0] == 0 or boxes_b.shape[0] == 0:
            return torch.empty((boxes_a.shape[0], boxes_b.shape[0]), dtype=torch.float32, device=self.device)

        tensor_a = torch.as_tensor(boxes_a, dtype=torch.float32, device=self.device)
        tensor_b = torch.as_tensor(boxes_b, dtype=torch.float32, device=self.device)
        merged = torch.cat([tensor_a, tensor_b], dim=0)
        prob_matrix = batch_prob_iou(merged)
        split = tensor_a.shape[0]
        return prob_matrix[:split, split:]


def build_obb_iou_backend(cfg=None, device=None):
    backend_name = resolve_obb_iou_backend_name(cfg)
    if backend_name == OBB_IOU_BACKEND_CPU_POLYGON:
        return CPUPolygonIoUBackend()
    if backend_name == OBB_IOU_BACKEND_GPU_PROB:
        return GPUProbIoUBackend(device=device)
    raise ValueError(
        f"Unsupported OBB IoU backend '{backend_name}'. "
        f"Supported backends: {OBB_IOU_BACKEND_CPU_POLYGON}, {OBB_IOU_BACKEND_GPU_PROB}."
    )
