# Experiment Plan

## Core Experiments

1. `baseline`
2. `fusion_main`
3. `assigner_main`
4. `temporal_main`
5. `full_project`

## Recommended Order

1. Run `baseline` first to establish the common control setup.
2. Run `fusion_main` to measure RGB-T complementarity under all-weather conditions.
3. Run `assigner_main` to measure gains on tiny rotated targets.
4. Run `temporal_main` to validate sequence stability.
5. Run `full_project` last to confirm the three mainlines can stack.

## What To Watch

### baseline

- Main metrics: `mAP_50`, `mAP_50_95`
- Small-object metrics: `mAP_S`, `Recall_S`

### fusion_main

- Main metrics: `mAP_50`
- Grouped metrics: `time_of_day`, `weather`
- Robustness metrics: `RGBDrop`, `IRDrop`

### assigner_main

- Main metrics: `mAP_S`, `Recall_S`
- Error-analysis focus: tiny FN, elongated FN, confusion summary

### temporal_main

- Main metric: `TemporalStability`
- Analysis focus: night / low-texture misses and flicker reduction

### full_project

- Main metrics: `mAP_50`, `mAP_S`, `TemporalStability`
- Analysis focus: grouped metrics + error analysis summary

## Recommended Outputs To Record

- main metrics dict
- grouped metrics
- error analysis summary
- per-image error CSV / JSON
- representative visualization samples

## Config Entry

- `configs/exp_baseline.yaml`
- `configs/exp_fusion_main.yaml`
- `configs/exp_assigner_main.yaml`
- `configs/exp_temporal_main.yaml`
- `configs/exp_full_project.yaml`

## Compatibility Note

- Legacy configs still load.
- New configs are the recommended main entry for future training and reporting.
