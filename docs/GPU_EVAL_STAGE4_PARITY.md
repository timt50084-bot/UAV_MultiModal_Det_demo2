# GPU Eval Stage 4

## Goal

Stage 4 added a repeatable CPU-vs-GPU comparison tool for the detection evaluator so drift and runtime could be measured before any default switch was considered.
After Stage 5 switched the CUDA-capable detection default to GPU, this compare tool remains the recommended regression guard.

## Comparison Entry

Use:

```bash
python tools/compare_detection_evaluators.py --config configs/main/full_project.yaml --weights <checkpoint> --device 0
```

The tool always compares the same checkpoint and validation set with two fixed combinations:

- CPU reference:
  - `eval.evaluator=cpu`
  - `eval.obb_iou_backend=cpu_polygon`
- GPU mainline:
  - `eval.evaluator=gpu`
  - `eval.obb_iou_backend=gpu_prob`

## Why Parity Is Not "Exact Equality"

The CPU reference path uses exact polygon IoU through shapely.
The GPU path uses `gpu_prob`, which is a ProbIoU surrogate similarity.

That means Stage 4 does **not** treat CPU and GPU outputs as numerically identical by definition.
The parity goal is instead:

- quantify drift
- quantify runtime benefit
- flag unusual regressions
- preserve the CPU evaluator as the reference baseline

## Comparison Scope

Stage 4 intentionally isolates the base detection evaluator path.

Compared metrics:

- `mAP_50`
- `mAP_50_95`
- `Precision`
- `Recall`
- `mAP_S`

Not included in the Stage 4 parity gate:

- `Recall_S`
- `Precision_S`
- `TemporalStability`
- `GroupedMetrics`
- `ErrorAnalysis`
- tracking metrics

To keep the comparison focused and fair, the compare tool disables:

- `cross_modal_robustness`
- `temporal_stability`
- `group_eval`
- `error_analysis`

for both runs.

## Output Artifacts

The tool writes:

- `detection_evaluator_compare.json`
- `detection_evaluator_compare.md`

The JSON contains:

- run metadata
- CPU metrics
- GPU metrics
- absolute drift
- relative drift
- runtime summary
- advisory / strict gate results

The Markdown report is the human-readable companion for review and archiving.

## Gate Design

The default gate is advisory, not strict.

Why:

- the GPU path is a surrogate evaluator, not an exact polygon-IoU clone
- early parity work should surface suspicious drift without blocking every run

Current thresholds are deliberately conservative and are used to detect obvious regressions, not to claim full equivalence.

`--strict-gate` can be enabled when a team wants CI-style failure behavior.

## How Stage 5 Uses This Tool

The default switch to `eval.evaluator=gpu` should be backed by evidence for all of the following:

1. Stable parity runs across representative checkpoints, not just one sample run.
2. Acceptable drift envelopes for:
   - `mAP_50`
   - `mAP_50_95`
   - `Precision`
   - `Recall`
   - `mAP_S`
3. Consistent runtime benefit on the real validation workload.
4. Clear documentation of where the GPU path still differs from the CPU reference path.
5. A regression workflow that can detect when later code changes worsen either drift or runtime.

After the default switch, the CPU evaluator should still remain the reference implementation for explicit fallback and parity checks.
