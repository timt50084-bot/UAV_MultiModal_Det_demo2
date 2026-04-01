from copy import deepcopy
from pathlib import Path
import warnings

from omegaconf import OmegaConf

from .analysis import TrackingErrorAnalyzer
from .io import export_tracking_artifacts, load_tracking_json
from .metrics import evaluate_mot_dataset, normalize_tracking_sequences
from .visualization import render_tracking_sequence


TRACKING_EVAL_DEFAULTS = {
    'enabled': False,
    'mot_metrics': True,
    'error_analysis': True,
    'export_results': True,
    'save_visualizations': False,
    'export_json': True,
    'export_csv': True,
    'export_mot_txt': False,
    'output_dir': 'outputs/tracking_eval',
    'results_path': '',
    'gt_path': '',
    'image_root': '',
    'small_object_area_threshold': 32,
    'long_track_min_length': 3,
    'matching': {
        'iou_threshold': 0.5,
        'use_obb_iou': True,
        'class_aware': True,
    },
    'grouped_analysis': {
        'enabled': True,
        'keys': ['time_of_day', 'weather', 'size_group'],
    },
    'visualization': {
        'enabled': True,
        'draw_trails': True,
        'trail_length': 20,
        'show_state': True,
    },
}


def normalize_tracking_eval_cfg(tracking_eval_cfg=None):
    cfg = deepcopy(TRACKING_EVAL_DEFAULTS)
    if tracking_eval_cfg is None:
        return cfg
    if OmegaConf.is_config(tracking_eval_cfg):
        tracking_eval_cfg = OmegaConf.to_container(tracking_eval_cfg, resolve=True)
    if not isinstance(tracking_eval_cfg, dict):
        return cfg
    requested_mot_metrics = tracking_eval_cfg.get('mot_metrics', TRACKING_EVAL_DEFAULTS['mot_metrics'])
    for key, value in tracking_eval_cfg.items():
        if key in {'matching', 'grouped_analysis', 'visualization'} and isinstance(value, dict):
            cfg[key].update(value)
        else:
            cfg[key] = value
    if 'mot_metrics' in tracking_eval_cfg and not bool(requested_mot_metrics):
        warnings.warn(
            'tracking_eval.mot_metrics is retained for compatibility, but tracking evaluation always computes '
            'and exports MOT metrics when it runs. The flag is ignored and treated as True.',
            stacklevel=2,
        )
        cfg['mot_metrics'] = TRACKING_EVAL_DEFAULTS['mot_metrics']
    return cfg


class TrackingEvaluator:
    def __init__(self, tracking_eval_cfg=None, class_names=None):
        self.cfg = normalize_tracking_eval_cfg(tracking_eval_cfg)
        self.class_names = list(class_names or [])

    def _missing_gt_result(self, pred_sequences=None, reason='missing_tracking_gt'):
        analysis = {
            'available': False,
            'summary': {'available': False, 'reason': reason, 'grouped_analysis': {}},
            'per_sequence_analysis': [],
            'per_track_analysis': [],
        }
        if pred_sequences and self.cfg.get('error_analysis', True):
            analysis = TrackingErrorAnalyzer(self.cfg, class_names=self.class_names).analyze_runtime(pred_sequences)

        result = {
            'available': False,
            'reason': reason,
            'metrics': None,
            'analysis': analysis,
            'per_sequence_metrics': {},
            'exported_files': {},
        }
        if self.cfg.get('export_results', True):
            result['exported_files'] = export_tracking_artifacts(
                self.cfg.get('output_dir', 'outputs/tracking_eval'),
                metrics_summary={'available': False, 'reason': reason},
                analysis_result=result['analysis'],
                sequence_results=pred_sequences,
                export_json=bool(self.cfg.get('export_json', True)),
                export_csv=bool(self.cfg.get('export_csv', True)),
                export_mot_txt=bool(self.cfg.get('export_mot_txt', False)),
            )
        return result

    def _maybe_render_visualizations(self, pred_sequences):
        if not self.cfg.get('save_visualizations', False):
            return {}
        visualization_cfg = self.cfg.get('visualization', {}) if isinstance(self.cfg.get('visualization', {}), dict) else {}
        if not visualization_cfg.get('enabled', True):
            return {}
        image_root = self.cfg.get('image_root', '')
        if not image_root:
            return {}
        image_root = Path(image_root)
        if not image_root.exists():
            return {}

        visualization_dir = Path(self.cfg.get('output_dir', 'outputs/tracking_eval')) / 'visualizations'
        visualization_dir.mkdir(parents=True, exist_ok=True)
        saved = {}
        for sequence_id, sequence in pred_sequences.items():
            saved[sequence_id] = render_tracking_sequence(
                sequence,
                image_root=image_root,
                output_dir=visualization_dir / sequence_id.replace('/', '_'),
                class_names=self.class_names,
                visualization_cfg=visualization_cfg,
            )
        return saved

    def evaluate_from_sequences(self, pred_sequences, gt_sequences=None):
        pred_sequences = normalize_tracking_sequences(pred_sequences, kind='pred', default_sequence_id='pred')
        if not gt_sequences:
            result = self._missing_gt_result(pred_sequences=pred_sequences, reason='missing_tracking_gt')
            visualization_files = self._maybe_render_visualizations(pred_sequences)
            if visualization_files:
                result['exported_files']['visualizations'] = visualization_files
            return result

        gt_sequences = normalize_tracking_sequences(gt_sequences, kind='gt', default_sequence_id='gt')
        mot_result = evaluate_mot_dataset(pred_sequences, gt_sequences, config={'matching': self.cfg.get('matching', {})})
        if not mot_result.get('available', False):
            result = self._missing_gt_result(pred_sequences=pred_sequences, reason=mot_result.get('reason', 'missing_tracking_gt'))
            visualization_files = self._maybe_render_visualizations(pred_sequences)
            if visualization_files:
                result['exported_files']['visualizations'] = visualization_files
            return result

        analysis = TrackingErrorAnalyzer(self.cfg, class_names=self.class_names).analyze(mot_result) if self.cfg.get('error_analysis', True) else {
            'available': False,
            'summary': {'available': False, 'reason': 'disabled', 'grouped_analysis': {}},
            'per_sequence_analysis': [],
            'per_track_analysis': [],
        }

        exported_files = {}
        if self.cfg.get('export_results', True):
            exported_files = export_tracking_artifacts(
                self.cfg.get('output_dir', 'outputs/tracking_eval'),
                metrics_summary=mot_result.get('metrics'),
                analysis_result=analysis,
                sequence_results=pred_sequences,
                export_json=bool(self.cfg.get('export_json', True)),
                export_csv=bool(self.cfg.get('export_csv', True)),
                export_mot_txt=bool(self.cfg.get('export_mot_txt', False)),
            )

        visualization_files = self._maybe_render_visualizations(pred_sequences)
        if visualization_files:
            exported_files['visualizations'] = visualization_files

        return {
            'available': True,
            'reason': None,
            'metrics': mot_result.get('metrics'),
            'per_sequence_metrics': {sequence_id: sequence_result.get('metrics') for sequence_id, sequence_result in mot_result.get('per_sequence', {}).items()},
            'analysis': analysis,
            'exported_files': exported_files,
        }

    def evaluate_from_files(self, results_path, gt_path=None):
        pred_sequences = load_tracking_json(results_path)
        gt_sequences = load_tracking_json(gt_path) if gt_path else None
        return self.evaluate_from_sequences(pred_sequences, gt_sequences=gt_sequences)

    def evaluate_from_dataset(self, dataset):
        if dataset is None:
            return self._missing_gt_result(reason='missing_dataset')

        gt_sequences = None
        if hasattr(dataset, 'get_tracking_ground_truth'):
            gt_sequences = dataset.get_tracking_ground_truth()
        elif hasattr(dataset, 'tracking_ground_truth'):
            gt_sequences = getattr(dataset, 'tracking_ground_truth')
        elif hasattr(dataset, 'tracking_gt'):
            gt_sequences = getattr(dataset, 'tracking_gt')

        if gt_sequences is None:
            return self._missing_gt_result(reason='missing_tracking_gt')
        return self.evaluate_from_sequences(pred_sequences={}, gt_sequences=gt_sequences)

