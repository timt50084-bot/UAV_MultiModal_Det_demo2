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

- `configs/main/baseline.yaml`
- `configs/main/fusion_main.yaml`
- `configs/main/assigner_main.yaml`
- `configs/main/temporal_main.yaml`
- `configs/main/full_project.yaml`

## Compatibility Note

- Legacy `configs/exp_*.yaml` command lines still load through redirects.
- New `configs/main/*.yaml` files are the recommended main entry for future training and reporting.

## Tracking Mainline Note

- `configs/main/tracking_base.yaml` is the minimal tracking-by-detection entry.
- `configs/main/tracking_final.yaml` is the enhanced tracking mainline.
- `configs/main/tracking_eval.yaml` is the companion offline evaluation config for precomputed tracking outputs.
- Historical stage-by-stage tracking configs now live under `configs/archive/tracking/` and are kept only for old experiment reproduction or compatibility.
- Sequence tracking outputs are generated through `tools/infer.py` or `tools/track.py`.
- Recommended comparison metrics: `MOTA`, `IDF1`, `IDSwitches`, `Fragmentations`, and the small-object tracking summary.
- Recommended outputs: `mot_metrics.json`, `tracking_error_summary.json`, `per_sequence_tracking_analysis.json`, `per_track_analysis.csv`, and rendered sequence visualizations when image roots are available.
- If tracking ground truth is missing, the evaluation entry keeps a structured skip result instead of failing.

## Final Closure Workflow

### Recommended End-to-End Order

1. Detection training and validation
   - `baseline -> fusion_main -> assigner_main -> temporal_main -> full_project`
2. Tracking sequence inference and offline evaluation
   - `tracking_base -> tracking_final`
   - Run `tracking_eval` as the shared offline evaluation companion when you already have `tracking_results.json`.
3. Unified result summarization
   - `python tools/summarize_results.py --experiments-root outputs/experiments --output-dir outputs/summary`
4. Delivery material preparation
   - Fill `docs/ABLATION_TABLE_TEMPLATE.md`
   - Fill `docs/TECHNICAL_PLAN_TEMPLATE.md`
   - Adapt `docs/PPT_OUTLINE.md`
   - Adapt `docs/DEMO_SCRIPT.md`
   - Fill `docs/RESULTS_TRACKING_TEMPLATE.md`

### Final Tool Entry Points

- `tools/run_experiment_suite.py`
  - `--mode plan`: print the unified execution plan without launching experiments
  - `--mode train`: generate or execute detection training commands
  - `--mode eval`: generate or execute detection validation commands
  - `--mode track_eval`: generate tracking inference and offline tracking evaluation commands
- `tools/summarize_results.py`
  - collect detection / tracking results into unified CSV and JSON outputs

### Suggested Deliverables Checklist

- Detection main table ready
- Tracking main table ready
- Grouped analysis ready
- Visualization cases selected
- Ablation tables filled
- Technical plan skeleton filled
- PPT outline adapted
- Demo script adapted
