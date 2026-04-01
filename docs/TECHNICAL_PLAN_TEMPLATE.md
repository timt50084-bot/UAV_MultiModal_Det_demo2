# Technical Plan Template

## 1. Goal

- Project objective
- Expected deliverable
- Why this repo uses RGB+IR + OBB + tracking

## 2. Scope

- Detection scope
- Tracking scope
- Training / evaluation / report scope

## 3. Assumptions

- Dataset status
- Compute status
- Available grouped metadata
- Available tracking ground truth

## 4. End-to-End Pipeline

- Detector training path
- Tracking inference path
- Detection -> Tracking -> Eval -> Summary flow

## 5. Detection Plan

### 5.1 Baseline

- Baseline config
- Main metrics
- Expected risks

### 5.2 Mainline Components

- ReliabilityAwareFusion
- two_frame temporal refinement
- Tiny-aware / angle-aware assigner

### 5.3 Final Detector

- `configs/main/full_project.yaml`
- Main metrics
- Expected analysis outputs

## 6. Tracking Plan

- `configs/main/tracking_base.yaml`: minimal tracking-by-detection baseline
- `configs/main/tracking_final.yaml`: enhanced tracking mainline
- `configs/main/tracking_eval.yaml`: companion offline evaluation config
- Archived stage-by-stage tracking configs under `configs/archive/tracking/` are historical / compatibility-only and should be excluded from the default plan unless old experiments must be reproduced

## 7. Training / Validation

- Training schedule
- Validation schedule
- Checkpoint policy
- Main logging outputs

## 8. Evaluation

- Detection metrics
- Tracking metrics
- Grouped analysis
- Runtime / complexity metrics

## 9. Ablation Plan

- Detection ablation
- Tracking comparison: `tracking_base` vs `tracking_final`
- Add archived tracking stages only when reproduction is a hard requirement

## 10. Risks

- Data quality
- Runtime cost
- Missing annotations
- Tracking continuity / reactivation / overlap risks

## 11. Deliverables

- Main report outputs
- Detector / tracker result package
- Presentation materials
