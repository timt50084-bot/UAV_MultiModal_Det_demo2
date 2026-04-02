# GPU Eval Stage 6 Cleanup

## Goal

Stage 6 closes the detection-side GPU evaluator migration by removing CPU validation paths that only existed as transition scaffolding, while keeping the CPU pieces that still have a clear role.

This stage does **not**:

- change the default GPU detection mainline
- remove non-CUDA fallback
- remove the CPU side used by `tools/compare_detection_evaluators.py`
- remove CPU-only analysis islands such as ErrorAnalysis support
- touch tracking evaluation

## Cleanup Inventory

### A. CPU detection paths that were redundant and are now removed

1. CPU evaluator + `gpu_prob` mixed transition mode
   - This path let `eval.evaluator=cpu` run with `eval.obb_iou_backend=gpu_prob`.
   - It was no longer needed after the project gained:
     - a real GPU evaluator mainline
     - a dedicated compare tool
     - a non-CUDA fallback path
   - Keeping it would continue to imply that CPU and GPU were co-equal long-term detection mainlines.

2. Tests that explicitly endorsed the mixed CPU + `gpu_prob` path
   - Those tests were migration scaffolding.
   - They are replaced with tests that assert CPU now resolves back to the reference path.

### B. CPU detection paths that must stay

1. Explicit CPU reference mode
   - `eval.evaluator=cpu`
   - `eval.obb_iou_backend=cpu_polygon`
   - Role: exact shapely reference behavior

2. Non-CUDA fallback
   - Default requests GPU
   - Non-CUDA runtime warns and falls back to `cpu + cpu_polygon`
   - Role: compatibility and operational safety

3. Compare tool CPU side
   - `tools/compare_detection_evaluators.py` still needs `cpu + cpu_polygon`
   - Role: regression / parity baseline

### C. CPU analysis islands that still stay

1. `polygon_iou()` / `obb2polygon()` / `CPUPolygonIoUBackend`
   - Role: exact polygon reference implementation

2. `OBBMetricsEvaluator`
   - Role: CPU reference and fallback evaluator

3. ErrorAnalysis CPU path
   - Role: analysis-only island that still consumes CPU artifacts even when the GPU evaluator drives the base detection metrics

## What This Stage Deletes

- the mixed CPU evaluator + `gpu_prob` transition path
- the tests and wording that treated that path as supported behavior

## What This Stage Explicitly Does Not Delete

- explicit CPU reference mode
- non-CUDA automatic fallback
- compare tool CPU reference side
- CPU exact polygon IoU implementation
- CPU-only analysis support paths

## Resulting Detection Evaluator Roles

- GPU evaluator:
  - the only maintained detection mainline
- CPU evaluator:
  - reference mode
  - fallback mode
  - analysis support mode

The CPU evaluator is no longer described or tested as a parallel mainline path.
