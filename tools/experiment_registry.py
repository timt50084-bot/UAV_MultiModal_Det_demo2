"""Registry helpers for the active detection and tracking experiment suite.

The active tracking suite intentionally contains only tracking_base and
tracking_final. tracking_eval remains a companion offline-evaluation config and
is not listed as a runnable train/infer experiment here.
"""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    category: str
    config: str
    stage: str
    description: str


DETECTION_EXPERIMENTS = (
    ExperimentSpec(
        'baseline',
        'detection',
        'configs/main/baseline.yaml',
        'detection_baseline',
        'Baseline detector: SimpleConcatFusion with temporal enhancement disabled.',
    ),
    ExperimentSpec(
        'fusion_main',
        'detection',
        'configs/main/fusion_main.yaml',
        'detection_fusion',
        'Detector ablation with ReliabilityAwareFusion enabled.',
    ),
    ExperimentSpec(
        'assigner_main',
        'detection',
        'configs/main/assigner_main.yaml',
        'detection_assigner',
        'Detector ablation with tiny-aware / angle-aware assigner enabled.',
    ),
    ExperimentSpec(
        'temporal_main',
        'detection',
        'configs/main/temporal_main.yaml',
        'detection_temporal',
        'Detector ablation with two-frame temporal refinement enabled.',
    ),
    ExperimentSpec(
        'full_project',
        'detection',
        'configs/main/full_project.yaml',
        'detection_full',
        'Detector mainline: ReliabilityAwareFusion + two_frame temporal + assigner enhancements.',
    ),
)

TRACKING_EXPERIMENTS = (
    ExperimentSpec(
        'tracking_base',
        'tracking',
        'configs/main/tracking_base.yaml',
        'tracking_base',
        'Tracking baseline: minimal tracking-by-detection entry.',
    ),
    ExperimentSpec(
        'tracking_final',
        'tracking',
        'configs/main/tracking_final.yaml',
        'tracking_final',
        'Tracking mainline: enhanced tracking-by-detection final route.',
    ),
)

ALL_EXPERIMENTS = DETECTION_EXPERIMENTS + TRACKING_EXPERIMENTS


def get_experiments(subset: str = 'all') -> List[ExperimentSpec]:
    subset = str(subset or 'all').lower()
    if subset == 'detection':
        return list(DETECTION_EXPERIMENTS)
    if subset == 'tracking':
        return list(TRACKING_EXPERIMENTS)
    return list(ALL_EXPERIMENTS)


def recommended_order() -> List[str]:
    return [spec.name for spec in ALL_EXPERIMENTS]


def experiments_as_dicts(subset: str = 'all'):
    return [asdict(spec) for spec in get_experiments(subset=subset)]


def infer_run_root(spec: ExperimentSpec, experiments_root: str = 'outputs/experiments') -> Path:
    return Path(experiments_root) / spec.name


def infer_detection_weights(spec: ExperimentSpec, experiments_root: str = 'outputs/experiments') -> Path:
    return infer_run_root(spec, experiments_root=experiments_root) / 'weights' / 'best.pt'


def infer_tracking_detector_weights(
    experiments_root: str = 'outputs/experiments',
    detector_run: str = 'full_project',
) -> Path:
    return Path(experiments_root) / detector_run / 'weights' / 'best.pt'


def infer_tracking_output_dir(spec: ExperimentSpec, experiments_root: str = 'outputs/experiments') -> Path:
    return infer_run_root(spec, experiments_root=experiments_root) / 'tracking_infer'


def infer_tracking_results_path(spec: ExperimentSpec, experiments_root: str = 'outputs/experiments') -> Path:
    return infer_tracking_output_dir(spec, experiments_root=experiments_root) / 'tracking_results.json'


def infer_tracking_eval_dir(spec: ExperimentSpec, experiments_root: str = 'outputs/experiments') -> Path:
    return infer_run_root(spec, experiments_root=experiments_root) / 'tracking_eval'
