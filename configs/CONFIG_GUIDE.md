# Config Guide

## Recommended daily entry

- Base defaults live in `configs/default.yaml`.
- The only recommended day-to-day full-project training entry is `configs/main/full_project.yaml`.
- `configs/exp_full_project.yaml` is now only a backward-compatible wrapper.

## Where to edit common training params

Edit these in `configs/main/full_project.yaml`:

- `dataloader.batch_size`
- `dataloader.num_workers`
- `dataloader.use_log_sampler`
- `dataset.imgsz`
- `train.epochs`
- `train.lr`
- `train.weight_decay`
- `train.accumulate`
- `performance.profile_train`
- `performance.dataloader.persistent_workers`
- `performance.dataloader.prefetch_factor`

Most common speed / schedule tuning is intentionally concentrated near the top of that file.

## Directory layout

- `configs/default.yaml`
  Base defaults shared across training, validation, inference, and tracking.
- `configs/main/`
  Main training entry points.
  `full_project.yaml` is the recommended detector training config.
- `configs/ablation/`
  Detector-side ablations and historical experiment configs such as baseline, fusion, assigner, and temporal variants.
- `configs/tracking/`
  Tracking-specific experiment configs such as base, temporal, modality, jointlite, and final.
- `configs/eval/`
  Evaluation-only configs such as full eval, error analysis, and task metrics.
- `configs/infer/`
  Inference-only configs such as competition inference.
- `configs/dataset/`, `configs/model/`
  Reusable config fragments and references. These are usually not edited for daily runs.

## Backward compatibility

- Older root-level paths such as `configs/exp_baseline.yaml` and `configs/exp_tracking_final.yaml` still work.
- The loader redirects those old paths to the new categorized files and prints a warning.
- Legacy top-level fields like `batch_size`, `num_workers`, and `use_log_sampler` are still accepted, but startup warns and maps them into `dataloader.*`.

## What not to edit for daily training

- Do not use files under `configs/ablation/` for normal full-project runs unless you are intentionally running an ablation.
- Do not edit files under `configs/dataset/` or `configs/model/` for routine hyperparameter changes.
- Do not edit `configs/exp_full_project.yaml` for daily work. It forwards to `configs/main/full_project.yaml`.

## Common examples

If you want to change batch size, workers, image size, and epochs for a normal full-project run:

1. Open `configs/main/full_project.yaml`.
2. Change:
   - `dataloader.batch_size`
   - `dataloader.num_workers`
   - `dataset.imgsz`
   - `train.epochs`
3. Run training.
4. Confirm the printed effective config summary and inspect:

`outputs/experiments/<run_name>/resolved_config.yaml`

## Tracking entry points

Use files under `configs/tracking/`:

- `configs/tracking/base.yaml`
- `configs/tracking/temporal.yaml`
- `configs/tracking/modality.yaml`
- `configs/tracking/jointlite.yaml`
- `configs/tracking/final.yaml`

For backward compatibility, the old `configs/exp_tracking_*.yaml` command lines still resolve to these files.
