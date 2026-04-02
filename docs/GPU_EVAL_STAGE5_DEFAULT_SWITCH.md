# GPU Eval Stage 5

## Goal

Stage 5 promotes the existing GPU detection evaluator to the default detection path while keeping the CPU evaluator intact as the reference and fallback implementation.

This stage does **not**:

- delete `cpu_polygon`
- delete shapely-based polygon IoU
- change tracking evaluation
- claim that `gpu_prob` is exact polygon IoU

## New Default Behavior

Detection validation and training-time validation now default to:

- `eval.evaluator: gpu`
- `eval.obb_iou_backend: gpu_prob`

That means CUDA-capable detection runs prefer:

- `GPUDetectionEvaluator`
- `GPUOBBMetricsEvaluator`
- `gpu_prob` ProbIoU surrogate similarity

## CPU Reference Path Still Exists

The explicit CPU reference path remains:

```yaml
eval:
  evaluator: cpu
  obb_iou_backend: cpu_polygon
```

Use that combination when you need:

- the shapely exact polygon IoU reference path
- a fallback on CUDA-related issues
- a parity / regression baseline for `tools/compare_detection_evaluators.py`

In the current Stage 6 state, that CPU path is no longer treated as a parallel detection mainline.
It is kept only as:

- the explicit reference mode
- the non-CUDA fallback mode
- the CPU-side support path for still-unmigrated analysis features

## Non-CUDA Safety Strategy

Stage 5 uses compatibility-first fallback behavior.

If detection validation requests the GPU evaluator but the active runtime device is not CUDA:

1. the evaluator factory emits a warning
2. the run safely falls back to:
   - `eval.evaluator=cpu`
   - `eval.obb_iou_backend=cpu_polygon`

This keeps the default mainline usable in:

- `--device -1` CPU validation
- CPU-only development environments
- lightweight CI jobs without CUDA

Invalid semantic combinations still fail clearly:

- `eval.evaluator=gpu` + `eval.obb_iou_backend=cpu_polygon` -> error

If `eval.evaluator=cpu` is paired with any non-reference OBB IoU backend, the runtime now coerces it back to `cpu_polygon` with a warning.

## Important Metric Semantics

The new default GPU path uses `gpu_prob`.

That backend is:

- CUDA-backed
- useful for faster base detection validation
- a surrogate similarity

It is **not**:

- exact polygon IoU
- numerically identical to the CPU shapely reference path by definition

The CPU reference path therefore remains necessary for:

- exact reference checks
- long-term parity tracking
- CPU-only fallback and analysis support after Stage 6 cleanup

## Regression Guard

Use the Stage 4 compare tool to keep the default switch honest:

```bash
python tools/compare_detection_evaluators.py \
  --config configs/main/full_project.yaml \
  --weights <checkpoint> \
  --device 0
```

That tool still compares:

- `cpu + cpu_polygon`
- `gpu + gpu_prob`

and records:

- base detection metric drift
- runtime delta
- advisory / strict gate status

## Stage 6 Prerequisite

Before cleaning up any CPU validation code in Stage 6, the project should still confirm:

1. the GPU default remains stable across representative checkpoints
2. the compare tool continues to show controlled drift and runtime gain
3. explicit CPU fallback is still usable when needed
4. any remaining CPU-only analysis paths are either retained intentionally or replaced with an equivalent plan
