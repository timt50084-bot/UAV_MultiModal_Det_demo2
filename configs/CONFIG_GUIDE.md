# Config Guide

## Recommended daily entry

- Base defaults live in `configs/default.yaml`.
- The only recommended day-to-day full-project training entry is `configs/main/full_project.yaml`.
- For tracking day-to-day work, use `configs/main/tracking_base.yaml` or `configs/main/tracking_final.yaml`.
- Use `configs/main/tracking_eval.yaml` only as the companion offline-evaluation config for precomputed tracking results.
- Old `configs/exp_*.yaml` command lines are still accepted through loader redirects.

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
The detector ablation files under `configs/main/` inherit these common settings, so you usually do not need to repeat batch size, image size, or epoch changes in multiple places.

## Directory layout

- `configs/default.yaml`
  Base defaults shared across training, validation, inference, and tracking.
- `configs/main/`
  Main active entry points.
  `full_project.yaml` is the recommended detector training config.
  `baseline.yaml`, `fusion_main.yaml`, `assigner_main.yaml`, and `temporal_main.yaml` are the detector ablation entry points.
  `tracking_base.yaml` and `tracking_final.yaml` are the active tracking mainline configs.
  `tracking_eval.yaml` is the companion offline tracking-evaluation config.
- `configs/archive/`
  Historical detector / eval / infer configs kept only for backward compatibility and reference.
  Archived tracking stage configs now live under `configs/archive/tracking/`.
- `configs/dataset/`, `configs/model/`
  Reusable config fragments and references. These are usually not edited for daily runs.

## Backward compatibility

- Older root-level paths such as `configs/exp_baseline.yaml` and `configs/exp_tracking_final.yaml` still work.
- The loader redirects those old paths to the new `configs/main/` or `configs/archive/` locations and prints a warning.
- Legacy top-level fields like `batch_size`, `num_workers`, and `use_log_sampler` are still accepted, but startup warns and maps them into `dataloader.*`.
- Legacy detection-side temporal-memory configs are still loadable for historical experiments and compatibility, but the maintained detector mainline uses `model.temporal.mode: two_frame`.
- Legacy `fusion_att_type` is still accepted for compatibility, but maintained configs should use `model.fusion.type`.

## What not to edit for daily training

- Do not use files under `configs/archive/` for normal day-to-day runs unless you are intentionally reproducing an old experiment.
- Do not edit files under `configs/dataset/` or `configs/model/` for routine hyperparameter changes.
- Do not chase older `configs/exp_*.yaml` paths for daily work. Edit the corresponding file under `configs/main/`.

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

## Detector ablation entry points

Use files under `configs/main/`:

- `configs/main/baseline.yaml`
  Baseline detector with fusion / temporal / tiny-angle-aware assigner enhancements disabled.
- `configs/main/fusion_main.yaml`
  Reliability-aware fusion enabled, temporal and assigner enhancements disabled.
- `configs/main/assigner_main.yaml`
  Tiny-angle-aware assigner enabled, fusion and temporal enhancements disabled.
- `configs/main/temporal_main.yaml`
  Two-frame temporal enabled, fusion and assigner enhancements disabled.
- `configs/main/full_project.yaml`
  Final detector mainline with `ReliabilityAwareFusion` + `two_frame` temporal + assigner enhancements all enabled.

Recommended workflow:

1. Change common training hyperparameters only in `configs/main/full_project.yaml`.
2. For a comparison run, switch the `--config` path to the corresponding file under `configs/main/`.
3. Keep the ablation wrapper focused on feature on/off choices, not on duplicated schedule settings.

## Tracking entry points

Use files under `configs/main/`:

- `configs/main/tracking_base.yaml`
- `configs/main/tracking_final.yaml`
- `configs/main/tracking_eval.yaml`
  Offline evaluation helper for existing tracking outputs, not a tracking mainline training / inference config.

Archived stage-by-stage tracking configs are available under `configs/archive/tracking/` for historical experiments only.
For backward compatibility, the old `configs/exp_tracking_*.yaml` command lines still resolve through redirects.
