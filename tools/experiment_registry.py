"""Registry helpers for the unified detection and tracking experiment suite."""

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
        'Stable baseline detector.',
    ),
    ExperimentSpec(
        'fusion_main',
        'detection',
        'configs/main/fusion_main.yaml',
        'detection_fusion',
        'Detection mainline with reliability-aware fusion.',
    ),
    ExperimentSpec(
        'assigner_main',
        'detection',
        'configs/main/assigner_main.yaml',
        'detection_assigner',
        'Detection mainline with tiny-aware / angle-aware assigner.',
    ),
    ExperimentSpec(
        'temporal_main',
        'detection',
        'configs/main/temporal_main.yaml',
        'detection_temporal',
        'Detection mainline with temporal memory enabled.',
    ),
    ExperimentSpec(
        'full_project',
        'detection',
        'configs/main/full_project.yaml',
        'detection_full',
        'Unified full detector configuration.',
    ),
)

TRACKING_EXPERIMENTS = (
    ExperimentSpec(
        'tracking_base',
        'tracking',
        'configs/main/tracking_base.yaml',
        'tracking_stage1',
        'Tracking stage 1: tracking-by-detection baseline.',
    ),
    ExperimentSpec(
        'tracking_assoc',
        'tracking',
        'configs/main/tracking_assoc.yaml',
        'tracking_stage2',
        'Tracking stage 2: appearance-aware association.',
    ),
    ExperimentSpec(
        'tracking_temporal',
        'tracking',
        'configs/main/tracking_temporal.yaml',
        'tracking_stage3',
        'Tracking stage 3: temporal memory tracking.',
    ),
    ExperimentSpec(
        'tracking_modality',
        'tracking',
        'configs/main/tracking_modality.yaml',
        'tracking_stage5',
        'Tracking stage 5: modality-aware dynamic association.',
    ),
    ExperimentSpec(
        'tracking_jointlite',
        'tracking',
        'configs/main/tracking_jointlite.yaml',
        'tracking_stage6',
        'Tracking stage 6: joint-lite detector-tracker refinement.',
    ),
    ExperimentSpec(
        'tracking_final',
        'tracking',
        'configs/main/tracking_final.yaml',
        'tracking_stage7',
        'Tracking stage 7: advanced detector-tracker collaboration final.',
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
