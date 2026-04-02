# Config Guide

## Recommended daily entry

- Base defaults live in `configs/default.yaml`.
- The only recommended day-to-day full-project training entry is `configs/main/full_project.yaml`.
- For tracking day-to-day work, use `configs/main/tracking_base.yaml` or `configs/main/tracking_final.yaml`.
- Use `configs/main/tracking_eval.yaml` only as the companion offline-evaluation config for precomputed tracking results.
- Old `configs/exp_*.yaml` command lines are still accepted through loader redirects.

## Where to edit common training params

Edit these in `configs/main/full_project.yaml`:

- `dataset.root_dir`
- `dataloader.batch_size`
- `dataloader.num_workers`
- `dataloader.use_log_sampler`
- `dataset.imgsz`
- `train.epochs`
- `train.patience`
- `train.eval_interval`
- `train.lr`
- `train.weight_decay`
- `train.accumulate`
- `train.use_amp`
- `performance.profile_train`
- `performance.dataloader.persistent_workers`
- `performance.dataloader.prefetch_factor`

Most common speed / schedule tuning is intentionally concentrated near the top of that file.
The detector ablation files under `configs/main/` inherit these common settings, so you usually do not need to repeat batch size, image size, or epoch changes in multiple places.

Current `full_project.yaml` mainline defaults:

- `dataset.root_dir: dataset/DroneVehicle_process`
- `dataloader.batch_size: 4`
- `dataset.imgsz: 1024`
- `train.epochs: 300`
- `train.patience: 50`
- `train.eval_interval: 5`
- `train.lr: 0.0003`
- `train.use_amp: False`
- `eval.evaluator: gpu`
- `eval.obb_iou_backend: gpu_prob`

If `1024` is too expensive on the current machine, keep the mainline default and override at runtime with `dataset.imgsz=960` or `dataset.imgsz=800`.
Dataset-relative paths are normalized against the repo root at dataloader build time, so `dataset/DroneVehicle_process` remains valid even when training is launched outside the repo root.
`train.eval_interval` controls auto-validation frequency during training; the final epoch always validates. Early stopping patience still counts training epochs and is checked on validation epochs.
Detection validation now defaults to `eval.evaluator=gpu` together with `eval.obb_iou_backend=gpu_prob` on CUDA-capable runs. That path uses the Stage 3 GPU evaluator for the base detection metrics path.
`gpu_prob` remains a ProbIoU surrogate rather than exact shapely polygon IoU. The CPU reference path is still supported and remains the parity / regression baseline:

- `eval.evaluator=cpu`
- `eval.obb_iou_backend=cpu_polygon`

GPU is now the only maintained detection mainline. The CPU evaluator is no longer a parallel mainline path; it is retained only for explicit reference runs, non-CUDA fallback, and CPU-only analysis compatibility.
When the active device is not CUDA, the detection evaluator factory prints a warning and safely falls back to the CPU reference path instead of failing the run. This keeps CPU-only validation, lightweight CI, and `--device -1` workflows usable after the default switch.
`tools/compare_detection_evaluators.py` remains the recommended regression guard after the default switch. It still compares `cpu + cpu_polygon` against `gpu + gpu_prob` explicitly on the same checkpoint and validation set.
To pin the reference path explicitly, override:

- `eval.evaluator=cpu`
- `eval.obb_iou_backend=cpu_polygon`

If `eval.evaluator=cpu` is paired with any non-reference OBB IoU backend, the runtime now coerces it back to `cpu_polygon` with a warning instead of preserving a mixed CPU/GPU transition mode.

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
   - `train.eval_interval`
3. Run training.
4. Confirm the printed effective config summary and inspect:

`outputs/experiments/<run_name>/resolved_config.yaml`

## Detector ablation entry points

Use files under `configs/main/`:

- `configs/main/baseline.yaml`
  Baseline detector with fusion / temporal / tiny-angle-aware assigner enhancements disabled. This is the minimum detection-side comparison entry.
- `configs/main/fusion_main.yaml`
  Reliability-aware fusion enabled, temporal and assigner enhancements disabled.
- `configs/main/assigner_main.yaml`
  Tiny-angle-aware assigner enabled, fusion and temporal enhancements disabled.
- `configs/main/temporal_main.yaml`
  Two-frame temporal enabled, fusion and assigner enhancements disabled.
- `configs/main/full_project.yaml`
  Final detector mainline with `ReliabilityAwareFusion` + `two_frame` temporal + assigner enhancements all enabled.

These detector entry points inherit the shared mainline dataset path and training schedule from `full_project.yaml` unless they explicitly override a field.

Recommended workflow:

1. Change common training hyperparameters only in `configs/main/full_project.yaml`.
2. For a comparison run, switch the `--config` path to the corresponding file under `configs/main/`.
3. Keep the ablation wrapper focused on feature on/off choices, not on duplicated schedule settings.

## Tracking entry points

Use files under `configs/main/`:

- `configs/main/tracking_base.yaml`
  Minimal tracking-by-detection comparison entry.
- `configs/main/tracking_final.yaml`
  Enhanced tracking-by-detection mainline entry.
- `configs/main/tracking_eval.yaml`
  Offline evaluation helper for existing tracking outputs, not a tracking mainline training / inference config.

These tracking entry points also inherit the shared dataset path and training defaults from `full_project.yaml` unless they explicitly override a field.

Archived stage-by-stage tracking configs are available under `configs/archive/tracking/` for historical experiments only.
For backward compatibility, the old `configs/exp_tracking_*.yaml` command lines still resolve through redirects.
