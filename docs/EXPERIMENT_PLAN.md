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

## Tracking Stage 1 Note

- `configs/exp_tracking_base.yaml` is the minimal tracking-by-detection entry.
- Stage 1 focuses on stable `track_id` assignment and short-term recovery.
- Sequence tracking outputs are generated through `tools/infer.py` or `tools/track.py`.
- MOT metrics, ReID, and multi-frame fusion are intentionally deferred to later tracking stages.

## Tracking Stage 4 Note

- Use `configs/exp_tracking_base.yaml`, `configs/exp_tracking_assoc.yaml`, `configs/exp_tracking_temporal.yaml`, `configs/exp_tracking_modality.yaml`, `configs/exp_tracking_jointlite.yaml`, and `configs/exp_tracking_final.yaml` to export comparable sequence results.
- Use `configs/exp_tracking_eval.yaml` for offline tracking evaluation and analysis.
- Recommended comparison metrics: `MOTA`, `IDF1`, `IDSwitches`, `Fragmentations`, and the small-object tracking summary.
- Recommended outputs: `mot_metrics.json`, `tracking_error_summary.json`, `per_sequence_tracking_analysis.json`, `per_track_analysis.csv`, and rendered sequence visualizations when image roots are available.
- If tracking ground truth is missing, the evaluation entry keeps a structured skip result instead of failing.

## Tracking Stage 5 Note

- `configs/exp_tracking_modality.yaml` is the stage-5 entry for modality-aware dynamic association and scene-adaptive tracking.
- Compare it directly with `exp_tracking_base`, `exp_tracking_assoc`, and `exp_tracking_temporal` to measure how modality awareness helps under night, fog, and low-visibility conditions.
- Key stage-5 counters to record: `rgb_dominant_association_count`, `ir_dominant_association_count`, `balanced_association_count`, `low_confidence_motion_fallback_count`, and `modality_helped_reactivation_count`.
- If grouped metadata is available, inspect `time_of_day` and `weather` splits for night / fog robustness.

## Tracking Stage 6 Note

- `configs/exp_tracking_jointlite.yaml` is the stage-6 entry for track-aware detection refinement.
- Compare it directly with `configs/exp_tracking_modality.yaml` to measure whether low-score rescue and track-guided prediction reduce short-term breaks on small and occluded targets.
- Key stage-6 counters to record: `rescued_detection_count`, `rescued_small_object_count`, `track_guided_prediction_count`, `predicted_only_track_count`, `refinement_helped_reactivation_count`, and `refinement_suppressed_false_drop_count`.
- If tracking ground truth is incomplete, keep the runtime analysis and visualization outputs as the first-pass validation path.

## Tracking Stage 7 Note

- `configs/exp_tracking_final.yaml` is the final tracking mainline for stronger detector-tracker collaboration and advanced temporal recovery.
- Compare it directly with `configs/exp_tracking_jointlite.yaml` to measure whether reactivation, overlap handling, and long-track continuity improve further.
- Key stage-7 counters to record: `feature_assist_reactivation_count`, `memory_reactivation_count`, `overlap_disambiguation_count`, `overlap_disambiguation_helped_count`, `long_track_continuity_score`, and `small_object_track_survival_rate`.
- If tracking ground truth is still incomplete, preserve runtime advanced summary outputs and qualitative tracking videos for the report package.

## Final Closure Workflow

### Recommended End-to-End Order

1. Detection training and validation
   - `baseline -> fusion_main -> assigner_main -> temporal_main -> full_project`
2. Tracking sequence inference and offline evaluation
   - `tracking_base -> tracking_assoc -> tracking_temporal -> tracking_modality -> tracking_jointlite -> tracking_final`
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