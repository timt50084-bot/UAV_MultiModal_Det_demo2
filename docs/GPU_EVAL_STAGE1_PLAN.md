# GPU Eval Stage 1

## Goal

Stage 1 does not replace the current CPU validation logic.
It only documents the real evaluation chain and inserts a thin detection-evaluator entry layer so that a future GPU evaluator can be mounted without rewriting the existing CPU path first.

## Current Evaluation Chain

### Detection validation entry points

1. `tools/val.py`
   - Builds `val_loader` and model weights.
   - Runs detection validation through the detection evaluator.
   - Optionally runs tracking evaluation afterward if `tracking_eval.enabled` is set.
2. `tools/train.py`
   - Builds the same detection evaluator for epoch-end validation during training.
   - Passes it into `Trainer`, which only depends on `evaluator.evaluate(model, epoch=...) -> metrics_dict`.

### Detection evaluator runtime path

1. `tools/val.py` / `tools/train.py`
2. `src/engine/evaluator_factory.py::build_detection_evaluator`
3. `src/engine/evaluator.py::Evaluator.evaluate`
4. `src/engine/evaluator.py::_run_eval_pass`
5. Model forward and `flatten_predictions`
6. `src/model/bbox_utils.py::non_max_suppression_obb`
   - This path already uses torch / `batch_prob_iou` for NMS-style suppression.
   - This is not the final metric-matching implementation.
7. `metrics_evaluator.add_batch(...)`
8. `src/metrics/obb_metrics.py::OBBMetricsEvaluator.get_full_metrics`

### Detection metric aggregation path

`OBBMetricsEvaluator.get_full_metrics()` calls `_compute_metrics_dict()`, which aggregates:

- Base detection metrics:
  - `mAP_50`
  - `mAP_50_95`
  - `Precision`
  - `Recall`
- Small-object metrics:
  - `mAP_S`
  - `Recall_S`
  - `Precision_S`
- Temporal stability:
  - `TemporalStability`
- Grouped metrics:
  - `GroupedMetrics`

If enabled in `Evaluator.evaluate()`, the detection path also layers on:

- Cross-modal robustness from `src/metrics/task_specific_metrics.py`
- Error analysis from `src/metrics/error_analysis.py`

## Where CPU OBB IoU Really Happens

### Detection metrics

The current CPU OBB IoU hot path is:

1. `src/metrics/obb_metrics.py::polygon_iou`
2. `src/metrics/obb_metrics.py::obb2polygon`
3. `shapely.geometry.Polygon`

Behavior:

- `polygon_iou` first does `hbb_prescreening(...)`.
- If the prescreen passes, it converts both OBBs into shapely polygons.
- It computes intersection / union on CPU via shapely.
- A `poly_cache` dict avoids rebuilding polygons repeatedly within one metric pass.

This same `polygon_iou` path is reused by:

- `_compute_detection_metrics(...)` inside `OBBMetricsEvaluator`
- Small-object matching via `match_iou_fn`
- Temporal-stability matching via `match_iou_fn`
- Detection error analysis fallback matching in `ErrorAnalyzer`

So the detection-side CPU/shapely bottleneck is centralized in `src/metrics/obb_metrics.py`, even though several higher-level metrics consume it.

### Tracking evaluation

Tracking evaluation does not use the detection `polygon_iou` / shapely path.

Tracking path:

1. `tools/val.py` optional tracking branch
2. `src/tracking/evaluator.py::TrackingEvaluator`
3. `src/tracking/metrics.py::evaluate_mot_dataset`
4. `src/tracking/metrics.py::evaluate_mot_sequence`
5. `src/tracking/metrics.py::match_tracking_frame`
6. `src/tracking/metrics.py::pairwise_tracking_iou`

`pairwise_tracking_iou(...)` uses:

- `_pairwise_hbb_iou(...)` when `use_obb_iou=False`
- `src/model/bbox_utils.py::batch_prob_iou(...)` when `use_obb_iou=True`

That means tracking evaluation is already on a different overlap backend than detection validation.

## Detection vs Tracking Sharing

### Shared

- Same `tools/val.py` CLI entry can invoke both.
- Both consume normalized config and emit summary dicts/files.
- Both ultimately operate on OBB-like box records.

### Not shared

- Detection validation uses `src/engine/evaluator.py`.
- Detection metrics use `src/metrics/obb_metrics.py` and shapely-based `polygon_iou`.
- Tracking evaluation uses `src/tracking/evaluator.py` and `src/tracking/metrics.py`.
- Tracking matching does not call detection `polygon_iou`.

Conclusion:

- There is no single common OBB metric backend today.
- The detection-side evaluator is the safer Stage 2 GPU migration target.
- Tracking should stay untouched in this stage.

## Detection Data Flow

Current detection validation data flow is:

1. Model outputs dense predictions.
2. `flatten_predictions(...)` flattens per-level outputs.
3. `non_max_suppression_obb(...)` filters/merges candidate boxes.
4. `Evaluator` converts predictions and GT into per-image records.
5. `OBBMetricsEvaluator.add_batch(...)` stores:
   - `preds`: `image_id`, `bbox`, `score`, `class`
   - `gts`: `image_id`, `bbox`, `class`
   - `image_metadata`
6. `_compute_detection_metrics(...)` performs class-wise greedy matching:
   - sort predictions by score
   - per prediction, scan unmatched GT in the same image
   - compute OBB IoU through `polygon_iou(...)`
   - mark TP/FP
7. AP / mAP / precision / recall are aggregated.
8. Small-object, temporal, grouped, and error-analysis layers reuse the same prediction/GT records, and several of them reuse the same IoU callback shape.

## Stage 1 Abstraction Added

Stage 1 adds:

- `src/engine/evaluator_factory.py`
  - `get_detection_evaluator_backend(eval_cfg)`
  - `build_detection_evaluator(...)`

Behavior:

- Default backend is `eval.evaluator: cpu`.
- The factory still builds the existing CPU objects:
  - `OBBMetricsEvaluator`
  - `Evaluator`
- `tools/train.py` and `tools/val.py` now go through this builder instead of directly wiring the CPU classes.

This creates a stable mounting point for Stage 2 without changing the underlying CPU metric logic.

## Recommended Stage 2 GPU Hook Point

The safest Stage 2 landing point is:

1. Keep `tools/train.py` / `tools/val.py` unchanged.
2. Extend `src/engine/evaluator_factory.py` to select `cpu` vs future `gpu`.
3. Introduce the GPU detection evaluator behind the same `evaluate(model, epoch)` interface.
4. Inside that future GPU evaluator, replace the detection-side metric matching backend at the `OBBMetricsEvaluator` / `polygon_iou` layer boundary, not at the CLI or trainer layer.

Why this layer is safest:

- Training and validation already depend only on the evaluator entry object.
- The current CPU path remains intact for CPU-vs-GPU parity checks.
- Tracking evaluation remains isolated and untouched.
- The highest-risk math change is confined to detection metric matching, where parity can be measured directly against the preserved CPU implementation.
