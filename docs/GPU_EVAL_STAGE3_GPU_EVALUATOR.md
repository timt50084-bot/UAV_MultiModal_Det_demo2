# GPU Eval Stage 3

## Goal

Stage 3 promotes the existing GPU overlap capability into an opt-in detection evaluator path.
Stage 5 later promoted that path to the default detection evaluator on CUDA-capable runs, while keeping the CPU reference path explicit and intact.

## Evaluator Routing

Detection validation now supports two evaluator modes through `src/engine/evaluator_factory.py`:

- `eval.evaluator: cpu`
  - Returns the existing `Evaluator + OBBMetricsEvaluator` path.
  - In the current Stage 6 state, this is no longer a parallel mainline. It remains only as the CPU reference / fallback path.
- `eval.evaluator: gpu`
  - Returns `GPUDetectionEvaluator + GPUOBBMetricsEvaluator`.
  - Requires `eval.obb_iou_backend: gpu_prob`.
  - Resolves to the GPU path on CUDA devices and falls back to the CPU reference path on non-CUDA devices in the current Stage 5 runtime.

Invalid combination handling is explicit:

- `eval.evaluator=gpu` + `eval.obb_iou_backend=cpu_polygon` -> error
- `eval.evaluator=gpu` on a non-CUDA device -> warning + automatic fallback to `cpu + cpu_polygon`

## Stage 3 GPU Evaluator Coverage

The Stage 3 GPU evaluator is intentionally narrow.

### Included in the GPU evaluator path

- detection prediction accumulation on CUDA
- class-wise greedy matching for the base detection path
- base AP / mAP aggregation:
  - `mAP_50`
  - `mAP_50_95`
  - `Precision`
  - `Recall`
- `mAP_S` when small-object evaluation is enabled

### Still outside the GPU evaluator core path

- `Recall_S`
- `Precision_S`
- `TemporalStability`
- `GroupedMetrics`

These are not emitted by `GPUOBBMetricsEvaluator` in Stage 3.

### CPU fallback still allowed

- `ErrorAnalysis`

If error analysis is enabled while using the GPU evaluator, the evaluator still snapshots CPU artifacts and runs the existing CPU reference analysis path afterward.
This keeps the base GPU evaluator usable without pretending that all downstream analysis has been GPU-aligned already.

## Why the GPU Evaluator Was Not Initially the Default

The Stage 3 GPU evaluator uses:

- `eval.obb_iou_backend: gpu_prob`
- `src/model/bbox_utils.py::batch_prob_iou(...)`

That backend is a ProbIoU surrogate similarity, not exact shapely polygon IoU.
So Stage 3 was only about establishing a real GPU evaluator route for the base detection metrics path.
It was not yet the parity-confirmed replacement for the CPU evaluator at that stage.

## Recommended Stage 4

Stage 4 should focus on CPU-vs-GPU evaluator alignment and regression safety:

1. Run the same checkpoints through both evaluator modes:
   - `cpu + cpu_polygon`
   - `gpu + gpu_prob`
2. Compare:
   - `mAP_50`
   - `mAP_50_95`
   - `Precision`
   - `Recall`
   - `mAP_S`
3. Record drift distributions rather than only average drift.
4. Keep error analysis and tracking out of the parity gate until the base detection metrics are characterized.
5. Only consider a default switch after parity thresholds and runtime gains are both documented.
