# Ablation Table Template

Use this template for the current mainline narrative only. Archived tracking
stage configs under `configs/archive/tracking/` can be added back when you are
explicitly reproducing historical experiments, but they are intentionally
omitted from the default table.

## 1. Detection Runs

| Experiment | Fusion | Assigner | Temporal | mAP_50 | mAP_50_95 | Precision | Recall | mAP_S | Recall_S | Precision_S | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | SimpleConcatFusion | Off | Off |  |  |  |  |  |  |  |  |
| fusion_main | ReliabilityAwareFusion | Off | Off |  |  |  |  |  |  |  |  |
| assigner_main | SimpleConcatFusion | On | Off |  |  |  |  |  |  |  |  |
| temporal_main | SimpleConcatFusion | Off | two_frame |  |  |  |  |  |  |  |  |
| full_project | ReliabilityAwareFusion | On | two_frame |  |  |  |  |  |  |  |  |

## 2. Tracking Runs

| Experiment | Appearance | Tracking Memory | Modality Aware | Refinement | Feature Assist | MOTA | IDF1 | IDSwitches | Fragmentations | long_track_continuity_score | small_object_track_survival_rate | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tracking_base | Off | Off | Off | Off | Off |  |  |  |  |  |  |  |
| tracking_final | On | On | On | On | On |  |  |  |  |  |  |  |

`configs/main/tracking_eval.yaml` is the companion offline-evaluation config
for tracking outputs from the runs above.

## 3. Grouped Analysis

| Experiment | Day | Night | Fog / Low Visibility | Occlusion | Small Object | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| full_project |  |  |  |  |  |  |
| tracking_final |  |  |  |  |  |  |

Suggested grouped slices:

- Detection: `time_of_day`, `weather`, `occlusion`, `size_group`
- Tracking: grouped tracking analysis and runtime advanced summary

## 4. Runtime / Complexity

| Experiment | Params | FPS | Latency | Infer Mode | Tracking Mode | Visualization Output | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| full_project |  |  |  |  |  |  |  |
| tracking_final |  |  |  |  |  |  |  |

## 5. Main Highlights

- Detection: `RGBDrop / IRDrop / CrossModalRobustness_* / TemporalStability`
- Tracking: `feature_assist_reactivation_count / overlap_disambiguation_count / rescued_detection_count`
- Add historical tracking-stage rows only when the report explicitly includes archived reproduction experiments.
