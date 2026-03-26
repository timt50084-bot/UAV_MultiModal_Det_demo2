# Ablation Table Template

?????????????????????

## 1. Detection ????

| Experiment | Fusion | Assigner | Temporal | mAP_50 | mAP_50_95 | Precision | Recall | mAP_S | Recall_S | Precision_S | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline |  |  |  |  |  |  |  |  |  |  |  |
| fusion_main |  |  |  |  |  |  |  |  |  |  |  |
| assigner_main |  |  |  |  |  |  |  |  |  |  |  |
| temporal_main |  |  |  |  |  |  |  |  |  |  |  |
| full_project |  |  |  |  |  |  |  |  |  |  |  |

## 2. Tracking ????

| Experiment | Appearance | Temporal Memory | Modality Aware | JointLite Refinement | Feature Assist | MOTA | IDF1 | IDSwitches | Fragmentations | long_track_continuity_score | small_object_track_survival_rate | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tracking_base |  |  |  |  |  |  |  |  |  |  |  |  |
| tracking_assoc |  |  |  |  |  |  |  |  |  |  |  |  |
| tracking_temporal |  |  |  |  |  |  |  |  |  |  |  |  |
| tracking_modality |  |  |  |  |  |  |  |  |  |  |  |  |
| tracking_jointlite |  |  |  |  |  |  |  |  |  |  |  |  |
| tracking_final |  |  |  |  |  |  |  |  |  |  |  |  |

## 3. ???????

| Experiment | Day | Night | Fog / Low Visibility | Occlusion | Small Object | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| full_project |  |  |  |  |  |  |
| tracking_final |  |  |  |  |  |  |

???
- Detection ?? grouped metrics ?? `time_of_day / weather / occlusion / size`?
- Tracking ?? grouped tracking analysis ? runtime advanced summary?

## 4. Runtime / Complexity ?

| Experiment | Params | FPS | Latency | Infer Mode | Tracking Mode | Visualization Output | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| full_project |  |  |  |  |  |  |  |
| tracking_final |  |  |  |  |  |  |  |

## 5. ??????

- Detection?`RGBDrop / IRDrop / CrossModalRobustness_* / TemporalStability`
- Tracking?`feature_assist_reactivation_count / overlap_disambiguation_count / rescued_detection_count`
- ????????????????????????
