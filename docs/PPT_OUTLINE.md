# PPT Outline

## 1. Problem

- UAV RGB+IR perception under day / night / low-visibility conditions
- OBB detection for rotated small targets
- Tracking-by-detection with stable IDs and reduced short-term breaks

## 2. Project Scope

- Detector: RGB+IR dual stream + OBB detection
- Tracking: base baseline and enhanced final route
- Evaluation: detection metrics, tracking metrics, grouped analysis, report outputs

## 3. System Pipeline

- Detector training / validation
- Sequence inference with tracking
- Offline tracking evaluation and result summarization

## 4. Detection Mainline

- RGB+IR dual-stream detector
- ReliabilityAwareFusion
- two_frame temporal refinement
- Tiny-aware / angle-aware assigner

## 5. Tracking Mainline

- `tracking_base`: minimal tracking-by-detection baseline
- `tracking_final`: enhanced route with appearance, tracking memory, modality awareness, refinement, and feature-assisted reactivation
- Archived stage-by-stage tracking configs are historical only and should not be the main presentation path

## 6. Results

- Detection metrics and small-object metrics
- Tracking metrics and continuity metrics
- Grouped analysis by time of day / weather / size
- Runtime / complexity snapshot

## 7. Visual Cases

- Detection visualizations
- Tracking trajectory visualizations
- Day / night / low-visibility examples
- Failure and recovery cases

## 8. Closing

- Main takeaways
- Remaining risks
- Next-step suggestions
