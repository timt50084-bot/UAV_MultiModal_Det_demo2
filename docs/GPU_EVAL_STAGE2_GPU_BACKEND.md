# GPU Eval Stage 2

## Goal

Stage 2 adds a pluggable detection-side OBB IoU backend at the metrics layer.
It does not change the default validation path, does not remove the CPU shapely reference implementation, and does not touch tracking evaluation.

## Backend Added

Detection metrics now support:

- `cpu_polygon`
  - Existing reference path.
  - Exact polygon IoU on CPU through shapely intersection / union.
- `gpu_prob`
  - New opt-in CUDA backend.
  - Built on the existing `src/model/bbox_utils.py::batch_prob_iou(...)`.
  - This is a ProbIoU-based surrogate similarity, not exact polygon IoU.

## Why `gpu_prob` Is the Stage 2 Choice

This repo already contains a torch-friendly OBB similarity implementation in `batch_prob_iou(...)`, and tracking/NMS paths already use it.
That makes it the lowest-risk Stage 2 GPU backend foundation:

- no new heavy geometry dependency
- no custom GPU polygon clipping implementation
- small detection-metrics-only change surface
- easy to keep the CPU shapely path intact for later parity checks

The tradeoff is explicit:

- `cpu_polygon` = exact polygon IoU reference
- `gpu_prob` = CUDA surrogate similarity

Stage 2 keeps that distinction visible in naming, config, and code comments on purpose.

## Wiring in Detection Metrics

The new backend dispatch lives in:

- `src/metrics/obb_iou_backend.py`

`OBBMetricsEvaluator` now reads:

- `eval.obb_iou_backend`

and builds one of the backends above.

Current detection-side consumers that now share this backend source are:

- base detection matching in `OBBMetricsEvaluator._compute_detection_metrics(...)`
- small-object matching through `match_iou_fn`
- temporal-stability matching through `match_iou_fn`
- error-analysis matching when `OBBMetricsEvaluator` passes a custom matcher

## Default Behavior

Default config remains:

- `eval.evaluator: cpu`
- `eval.obb_iou_backend: cpu_polygon`

So the mainline validation result path is still the exact CPU/shapely implementation.

`gpu_prob` only activates when explicitly requested, for example:

```yaml
eval:
  evaluator: cpu
  obb_iou_backend: gpu_prob
```

If `gpu_prob` is requested without CUDA, the code raises a clear runtime error instead of silently changing behavior.

## What Stage 2 Can Do Now

- Provide a real CUDA OBB similarity backend in detection metrics.
- Let detection metrics choose `cpu_polygon` vs `gpu_prob`.
- Keep the existing CPU exact implementation unchanged for reference.

## What Stage 2 Still Does Not Do

- It does not make `gpu_prob` the default.
- It does not claim exact polygon IoU parity.
- It does not rewrite the full evaluator, AP aggregation, or tracking metrics.
- It does not add a GPU polygon-intersection implementation.

## Recommended Stage 3

Stage 3 should keep the current evaluator/trainer entry points unchanged and focus on detection matching internals:

1. Continue using `src/engine/evaluator_factory.py` as the stable evaluator entry.
2. Extend the detection metric path so per-image or per-class matching can consume more of the backend matrix output directly.
3. Keep `cpu_polygon` and `gpu_prob` side by side for explicit CPU-vs-GPU comparison runs.
4. Delay any default switch until a dedicated parity-analysis stage confirms acceptable drift.
