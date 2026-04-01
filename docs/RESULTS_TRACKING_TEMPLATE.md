# Results Tracking Template

## 1. Detection Runs

| Run | Config | Weights | Main Metrics Ready | Grouped Metrics Ready | Error Analysis Ready | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| baseline | `configs/main/baseline.yaml` |  |  |  |  |  |
| fusion_main | `configs/main/fusion_main.yaml` |  |  |  |  |  |
| assigner_main | `configs/main/assigner_main.yaml` |  |  |  |  |  |
| temporal_main | `configs/main/temporal_main.yaml` |  |  |  |  |  |
| full_project | `configs/main/full_project.yaml` |  |  |  |  |  |

## 2. Tracking Runs

| Run | Config | Detector Weights | tracking_results.json | MOT Metrics Ready | Advanced Summary Ready | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| tracking_base | `configs/main/tracking_base.yaml` |  |  |  |  |  |
| tracking_final | `configs/main/tracking_final.yaml` |  |  |  |  |  |

`configs/main/tracking_eval.yaml` is the companion offline-evaluation config for these runs.
Archived stage configs under `configs/archive/tracking/` are omitted from this main template because they are historical / compatibility-only.

## 3. Summary Delivery Checklist
- [ ] `outputs/summary/detection_summary.csv`
- [ ] `outputs/summary/tracking_summary.csv`
- [ ] `outputs/summary/project_summary.json`
- [ ] `docs/ABLATION_TABLE_TEMPLATE.md` filled
- [ ] `docs/PPT_OUTLINE.md` adapted
- [ ] `docs/DEMO_SCRIPT.md` updated with final assets
