# Results Tracking Template

## 1. Detection Runs

| Run | Config | Weights | Main Metrics Ready | Grouped Metrics Ready | Error Analysis Ready | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| baseline | `configs/exp_baseline.yaml` |  |  |  |  |  |
| fusion_main | `configs/exp_fusion_main.yaml` |  |  |  |  |  |
| assigner_main | `configs/exp_assigner_main.yaml` |  |  |  |  |  |
| temporal_main | `configs/exp_temporal_main.yaml` |  |  |  |  |  |
| full_project | `configs/exp_full_project.yaml` |  |  |  |  |  |

## 2. Tracking Runs

| Run | Config | Detector Weights | tracking_results.json | MOT Metrics Ready | Advanced Summary Ready | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| tracking_base | `configs/exp_tracking_base.yaml` |  |  |  |  |  |
| tracking_assoc | `configs/exp_tracking_assoc.yaml` |  |  |  |  |  |
| tracking_temporal | `configs/exp_tracking_temporal.yaml` |  |  |  |  |  |
| tracking_modality | `configs/exp_tracking_modality.yaml` |  |  |  |  |  |
| tracking_jointlite | `configs/exp_tracking_jointlite.yaml` |  |  |  |  |  |
| tracking_final | `configs/exp_tracking_final.yaml` |  |  |  |  |  |

## 3. Summary Delivery Checklist
- [ ] `outputs/summary/detection_summary.csv`
- [ ] `outputs/summary/tracking_summary.csv`
- [ ] `outputs/summary/project_summary.json`
- [ ] `docs/ABLATION_TABLE_TEMPLATE.md` filled
- [ ] `docs/PPT_OUTLINE.md` adapted
- [ ] `docs/DEMO_SCRIPT.md` updated with final assets
