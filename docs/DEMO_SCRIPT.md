# Demo Script

## 1. Demo Flow

1. Introduce the project goal and dataset setting.
2. Compare detector baseline with the full detector mainline.
3. Compare `tracking_base` with `tracking_final`.
4. Show evaluation outputs, summary artifacts, and documentation deliverables.

## 2. Assets To Prepare

- RGB + IR sample sequences
- Detection visualization outputs
- `tracking_results.json` for the demo sequences
- Summary CSV / JSON outputs

## 3. Key Talking Points

- RGB+IR dual-stream perception
- OBB detection for rotated UAV targets
- Tracking-by-detection from baseline to enhanced final route
- `tracking_final` improvements: tracking memory, modality awareness, refinement, reactivation, and overlap handling

## 4. Result Slides

- Detection: baseline vs full_project
- Tracking: tracking_base vs tracking_final
- Evaluation: MOTA / IDF1 / IDSwitches / advanced runtime summary
- Summary: consolidated CSV / JSON outputs

## 5. Notes

- Keep the detector story aligned with `ReliabilityAwareFusion + two_frame temporal`.
- Keep the tracking story aligned with `tracking_base` and `tracking_final`.
- Historical stage-by-stage tracking configs are archived and should only be mentioned when explicitly discussing old experiment reproduction.
