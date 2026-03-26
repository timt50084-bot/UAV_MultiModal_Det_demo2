from .track import Track as TrackState
from .tracker import MultiObjectTracker


class OBBTrackletManager(MultiObjectTracker):
    """Compatibility wrapper for the legacy stage-0 tracking script."""

    def __init__(self, iou_threshold=0.2, max_age=15, momentum=0.8, **kwargs):
        tracking_cfg = {
            'enabled': True,
            'method': 'tracking_by_detection',
            'max_age': max_age,
            'min_hits': kwargs.get('min_hits', 1),
            'init_score_threshold': kwargs.get('init_score_threshold', 0.0),
            'match_iou_threshold': iou_threshold,
            'max_center_distance': kwargs.get('max_center_distance', 50.0),
            'use_class_constraint': kwargs.get('use_class_constraint', True),
            'use_kalman': kwargs.get('use_kalman', True),
            'angle_smoothing': kwargs.get('angle_smoothing', momentum),
            'keep_history': kwargs.get('keep_history', 20),
        }
        super().__init__(tracking_cfg=tracking_cfg)
        self.momentum = momentum

    def update(self, detections, frame_meta=None):
        super().update(detections, frame_meta=frame_meta)
        return self.get_active_tracks(only_recent=True)
