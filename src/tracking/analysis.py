from collections import defaultdict
from copy import deepcopy

from src.metrics.task_specific_metrics import is_small_bbox, resolve_small_area_threshold


class TrackingErrorAnalyzer:
    def __init__(self, tracking_eval_cfg=None, class_names=None):
        tracking_eval_cfg = tracking_eval_cfg or {}
        self.cfg = tracking_eval_cfg
        self.class_names = list(class_names or [])
        self.grouped_cfg = tracking_eval_cfg.get('grouped_analysis', {}) if isinstance(tracking_eval_cfg, dict) else {}
        self.small_area_threshold = resolve_small_area_threshold(tracking_eval_cfg.get('small_object_area_threshold', 32))
        self.long_track_min_length = int(tracking_eval_cfg.get('long_track_min_length', 3))

    def _class_name(self, class_id):
        class_id = int(class_id)
        if 0 <= class_id < len(self.class_names):
            return self.class_names[class_id]
        return f'class_{class_id}'

    def _is_small_track(self, track_stats):
        for area in track_stats.get('areas', []):
            side = max(float(area), 0.0) ** 0.5
            if is_small_bbox([0.0, 0.0, side, side, 0.0], self.small_area_threshold):
                return True
        return False

    def analyze_runtime(self, pred_sequences):
        grouped_analysis = {key: {} for key in self.grouped_cfg.get('keys', [])} if self.grouped_cfg.get('enabled', True) else {}
        per_sequence_analysis = []
        per_track_analysis = []
        summary = self._empty_summary(available=True, runtime_only=True)
        runtime_tracks = defaultdict(list)

        for sequence_id, sequence in (pred_sequences or {}).items():
            seq_summary = self._empty_runtime_frame_summary()
            frames = sequence.get('frames', []) if isinstance(sequence, dict) else []
            for frame in frames:
                frame_summary = dict(frame.get('advanced_summary', {}) or {})
                if not frame_summary:
                    frame_summary = self._summarize_runtime_from_results(frame.get('results', []))
                self._accumulate_runtime_summary(summary, frame_summary)
                self._accumulate_runtime_summary(seq_summary, frame_summary)

                metadata = frame.get('metadata', {}) or {}
                for key in grouped_analysis.keys():
                    group_value = metadata.get(key)
                    if group_value in (None, ''):
                        continue
                    bucket = grouped_analysis[key].setdefault(str(group_value), self._empty_group_bucket())
                    bucket['frames'] += 1
                    self._accumulate_runtime_summary(bucket, frame_summary)

                for result in frame.get('results', []) or []:
                    track_id = result.get('track_id')
                    if track_id is not None:
                        runtime_tracks[(sequence_id, int(track_id))].append((int(frame.get('frame_index', 0)), result))
                    per_track_analysis.append(
                        {
                            'sequence_id': sequence_id,
                            'track_id': None if track_id is None else int(track_id),
                            'class_id': int(result.get('class_id', -1)),
                            'class_name': self._class_name(result.get('class_id', -1)),
                            'state': result.get('state'),
                            'refinement_source': result.get('refinement_source'),
                            'reactivation_source': result.get('reactivation_source'),
                            'rescued_detection': bool(result.get('rescued_detection', False)),
                            'predicted_candidate': bool(result.get('predicted_candidate', False)),
                            'feature_assist_reactivation': bool(result.get('feature_assist_reactivation', False)),
                            'memory_reactivation': bool(result.get('memory_reactivation', False)),
                            'overlap_disambiguated': bool(result.get('overlap_disambiguated', False)),
                            'overlap_disambiguation_helped': bool(result.get('overlap_disambiguation_helped', False)),
                        }
                    )

            per_sequence_analysis.append({'sequence_id': sequence_id, **seq_summary, 'num_frames': len(frames)})

        self._finalize_runtime_track_metrics(summary, runtime_tracks)
        summary['modality_association_summary'] = {
            'rgb_dominant_association_count': int(summary['rgb_dominant_association_count']),
            'ir_dominant_association_count': int(summary['ir_dominant_association_count']),
            'balanced_association_count': int(summary['balanced_association_count']),
            'low_confidence_motion_fallback_count': int(summary['low_confidence_motion_fallback_count']),
            'modality_helped_reactivation_count': int(summary['modality_helped_reactivation_count']),
        }
        summary['grouped_analysis'] = deepcopy(grouped_analysis)
        return {
            'available': True,
            'summary': summary,
            'per_sequence_analysis': per_sequence_analysis,
            'per_track_analysis': per_track_analysis,
        }

    def analyze(self, tracking_eval_result):
        if not tracking_eval_result or not tracking_eval_result.get('available', False):
            reason = tracking_eval_result.get('reason', 'missing_tracking_gt') if isinstance(tracking_eval_result, dict) else 'missing_tracking_gt'
            return {
                'available': False,
                'summary': {'available': False, 'reason': reason, 'grouped_analysis': {}},
                'per_sequence_analysis': [],
                'per_track_analysis': [],
            }

        per_sequence_analysis = []
        per_track_analysis = []
        grouped_analysis = {key: {} for key in self.grouped_cfg.get('keys', [])} if self.grouped_cfg.get('enabled', True) else {}
        summary = self._empty_summary(available=True, runtime_only=False)

        long_track_ratios = []
        small_track_ratios = []

        for sequence_id, sequence_result in tracking_eval_result.get('per_sequence', {}).items():
            metrics = sequence_result.get('metrics', {})
            details = sequence_result.get('details', {})
            frame_summaries = details.get('frame_summaries', [])
            gt_track_stats = details.get('gt_track_stats', {})
            switch_frames_in_sequence = 0
            seq_summary = self._empty_runtime_frame_summary()

            for frame_summary in frame_summaries:
                switch_events = frame_summary.get('id_switch_events', [])
                if switch_events:
                    switch_frames_in_sequence += 1
                    summary['id_switch_frames'].append(
                        {
                            'sequence_id': sequence_id,
                            'frame_index': frame_summary['frame_index'],
                            'count': len(switch_events),
                            'image_id': frame_summary.get('image_id'),
                        }
                    )

                runtime_summary = dict(frame_summary.get('advanced_summary', {}) or {})
                if not runtime_summary:
                    runtime_summary = self._summarize_runtime_from_results(frame_summary.get('pred_results', []))
                self._accumulate_runtime_summary(summary, runtime_summary)
                self._accumulate_runtime_summary(seq_summary, runtime_summary)

                frame_rgb = 0
                frame_ir = 0
                frame_balanced = 0
                frame_low_conf = 0
                frame_helped_reactivation = 0
                for match in frame_summary.get('matches', []):
                    mode = match.get('association_mode')
                    if mode == 'rgb_dominant':
                        summary['rgb_dominant_association_count'] += 1
                        frame_rgb += 1
                    elif mode == 'ir_dominant':
                        summary['ir_dominant_association_count'] += 1
                        frame_ir += 1
                    elif mode == 'balanced':
                        summary['balanced_association_count'] += 1
                        frame_balanced += 1
                    if match.get('low_confidence_motion_fallback', False):
                        summary['low_confidence_motion_fallback_count'] += 1
                        frame_low_conf += 1
                    if match.get('modality_helped_reactivation', False):
                        summary['modality_helped_reactivation_count'] += 1
                        frame_helped_reactivation += 1

                metadata = frame_summary.get('metadata', {}) or {}
                for key in grouped_analysis.keys():
                    group_value = metadata.get(key)
                    if group_value in (None, ''):
                        continue
                    bucket = grouped_analysis[key].setdefault(str(group_value), self._empty_group_bucket())
                    bucket['frames'] += 1
                    bucket['id_switch_frames'] += int(bool(switch_events))
                    bucket['id_switch_events'] += len(switch_events)
                    bucket['unmatched_gt'] += len(frame_summary.get('unmatched_gt_track_ids', []))
                    bucket['unmatched_pred'] += len(frame_summary.get('unmatched_pred_track_ids', []))
                    bucket['rgb_dominant_association_count'] += frame_rgb
                    bucket['ir_dominant_association_count'] += frame_ir
                    bucket['balanced_association_count'] += frame_balanced
                    bucket['low_confidence_motion_fallback_count'] += frame_low_conf
                    bucket['modality_helped_reactivation_count'] += frame_helped_reactivation
                    self._accumulate_runtime_summary(bucket, runtime_summary)

            per_sequence_analysis.append(
                {
                    'sequence_id': sequence_id,
                    'MOTA': metrics.get('MOTA'),
                    'IDF1': metrics.get('IDF1'),
                    'IDSwitches': int(metrics.get('IDSwitches', 0)),
                    'Fragmentations': int(metrics.get('Fragmentations', 0)),
                    'RecoveredTracks': int(metrics.get('RecoveredTracks', 0)),
                    'RecoveryOpportunities': int(metrics.get('RecoveryOpportunities', 0)),
                    'id_switch_frames': int(switch_frames_in_sequence),
                    'num_frames': int(metrics.get('num_frames', 0)),
                    **seq_summary,
                }
            )

            summary['id_switch_count'] += int(metrics.get('IDSwitches', 0))
            summary['fragmented_tracks'] += int(metrics.get('Fragmentations', 0))
            summary['reactivated_tracks'] += int(metrics.get('RecoveredTracks', 0))
            summary['recovery_opportunities'] += int(metrics.get('RecoveryOpportunities', 0))

            for track_id, track_stats in gt_track_stats.items():
                matched_ratio = float(track_stats.get('matched_ratio', 0.0))
                total_frames = int(track_stats.get('total_frames', 0))
                is_small_track = self._is_small_track(track_stats)
                if total_frames >= self.long_track_min_length:
                    long_track_ratios.append(matched_ratio)
                if is_small_track:
                    small_track_ratios.append(matched_ratio)
                    summary['small_object_tracking']['track_count'] += 1
                    summary['small_object_tracking']['id_switches'] += int(track_stats.get('id_switches', 0))
                    summary['small_object_tracking']['fragmentations'] += int(track_stats.get('fragmentations', 0))

                per_track_analysis.append(
                    {
                        'sequence_id': sequence_id,
                        'gt_track_id': int(track_id),
                        'class_id': int(track_stats.get('class_id', -1)),
                        'class_name': self._class_name(track_stats.get('class_id', -1)),
                        'total_frames': total_frames,
                        'matched_frames': int(track_stats.get('matched_frames', 0)),
                        'matched_ratio': matched_ratio,
                        'id_switches': int(track_stats.get('id_switches', 0)),
                        'fragmentations': int(track_stats.get('fragmentations', 0)),
                        'reactivated': bool(track_stats.get('reactivated', False)),
                        'size_group': track_stats.get('size_group'),
                        'avg_area': float(sum(track_stats.get('areas', [])) / max(len(track_stats.get('areas', [])), 1)) if track_stats.get('areas') else 0.0,
                        'is_small_track': bool(is_small_track),
                    }
                )

        summary['reactivation_success_rate'] = float(summary['reactivated_tracks'] / max(summary['recovery_opportunities'], 1)) if summary['recovery_opportunities'] > 0 else None
        summary['long_track_continuity_score'] = float(sum(long_track_ratios) / len(long_track_ratios)) if long_track_ratios else None
        summary['small_object_track_survival_rate'] = float(sum(small_track_ratios) / len(small_track_ratios)) if small_track_ratios else None
        summary['modality_association_summary'] = {
            'rgb_dominant_association_count': int(summary['rgb_dominant_association_count']),
            'ir_dominant_association_count': int(summary['ir_dominant_association_count']),
            'balanced_association_count': int(summary['balanced_association_count']),
            'low_confidence_motion_fallback_count': int(summary['low_confidence_motion_fallback_count']),
            'modality_helped_reactivation_count': int(summary['modality_helped_reactivation_count']),
        }
        summary['grouped_analysis'] = deepcopy(grouped_analysis)
        summary['id_switch_frames'] = sorted(summary['id_switch_frames'], key=lambda item: (item['sequence_id'], item['frame_index']))
        return {
            'available': True,
            'summary': summary,
            'per_sequence_analysis': per_sequence_analysis,
            'per_track_analysis': per_track_analysis,
        }

    def _summarize_runtime_from_results(self, results):
        summary = self._empty_runtime_frame_summary()
        states = []
        tracks_by_id = {}
        for result in results or []:
            states.append(result.get('state'))
            if result.get('track_id') is not None:
                tracks_by_id[int(result['track_id'])] = result
            summary['rescued_detection_count'] += int(bool(result.get('rescued_detection', False)))
            summary['rescued_small_object_count'] += int(bool(result.get('rescued_small_object', False)))
            summary['track_guided_prediction_count'] += int(bool(result.get('predicted_candidate', False)))
            summary['predicted_only_track_count'] += int(bool(result.get('predicted_candidate', False)))
            summary['refinement_helped_reactivation_count'] += int(bool(result.get('refinement_helped_reactivation', False)))
            summary['feature_assist_reactivation_count'] += int(bool(result.get('feature_assist_reactivation', False)))
            summary['memory_reactivation_count'] += int(bool(result.get('memory_reactivation', False)))
            summary['overlap_disambiguation_count'] += int(bool(result.get('overlap_disambiguated', False)))
            summary['overlap_disambiguation_helped_count'] += int(bool(result.get('overlap_disambiguation_helped', False)))
            summary['predicted_only_to_tracked_count'] += int(bool(result.get('predicted_only_to_tracked', False)))
            summary['predicted_candidate_reactivation_count'] += int(bool(result.get('predicted_candidate_reactivation', False)))
        summary['reactivating_state_count'] = sum(1 for state in states if state == 'reactivating')
        summary['refinement_suppressed_false_drop_count'] = int(summary['rescued_detection_count'] + summary['track_guided_prediction_count'])
        return summary

    def _accumulate_runtime_summary(self, target, source):
        for key in self._empty_runtime_frame_summary().keys():
            target[key] = int(target.get(key, 0)) + int(source.get(key, 0))

    def _finalize_runtime_track_metrics(self, summary, runtime_tracks):
        continuity = []
        small_survival = []
        for (_, track_id), items in runtime_tracks.items():
            items = sorted(items, key=lambda item: item[0])
            span = max(items[-1][0] - items[0][0] + 1, 1)
            observed = len(items)
            ratio = float(observed / span)
            if observed >= self.long_track_min_length:
                continuity.append(ratio)
            avg_area = 0.0
            small_votes = []
            for _, result in items:
                obb = result.get('obb')
                if obb is None:
                    obb = [0, 0, 0, 0, 0]
                area = max(float(obb[2]), 0.0) * max(float(obb[3]), 0.0)
                avg_area += area
                side = area ** 0.5
                small_votes.append(is_small_bbox([0.0, 0.0, side, side, 0.0], self.small_area_threshold))
            avg_area /= max(len(items), 1)
            if small_votes and any(small_votes):
                small_survival.append(ratio)
        summary['long_track_continuity_score'] = float(sum(continuity) / len(continuity)) if continuity else None
        summary['small_object_track_survival_rate'] = float(sum(small_survival) / len(small_survival)) if small_survival else None

    def _empty_summary(self, available=True, runtime_only=False):
        return {
            'available': available,
            'runtime_only': runtime_only,
            'id_switch_count': 0,
            'fragmented_tracks': 0,
            'reactivated_tracks': 0,
            'recovery_opportunities': 0,
            'reactivation_success_rate': None,
            'small_object_tracking': {
                'track_count': 0,
                'id_switches': 0,
                'fragmentations': 0,
            },
            'rgb_dominant_association_count': 0,
            'ir_dominant_association_count': 0,
            'balanced_association_count': 0,
            'low_confidence_motion_fallback_count': 0,
            'modality_helped_reactivation_count': 0,
            'rescued_detection_count': 0,
            'rescued_small_object_count': 0,
            'track_guided_prediction_count': 0,
            'predicted_only_track_count': 0,
            'refinement_helped_reactivation_count': 0,
            'refinement_suppressed_false_drop_count': 0,
            'feature_assist_reactivation_count': 0,
            'memory_reactivation_count': 0,
            'overlap_disambiguation_count': 0,
            'overlap_disambiguation_helped_count': 0,
            'reactivating_state_count': 0,
            'predicted_only_to_tracked_count': 0,
            'predicted_candidate_reactivation_count': 0,
            'long_track_continuity_score': None,
            'small_object_track_survival_rate': None,
            'modality_association_summary': {},
            'id_switch_frames': [],
            'grouped_analysis': {},
        }

    def _empty_runtime_frame_summary(self):
        return {
            'rescued_detection_count': 0,
            'rescued_small_object_count': 0,
            'track_guided_prediction_count': 0,
            'predicted_only_track_count': 0,
            'refinement_helped_reactivation_count': 0,
            'refinement_suppressed_false_drop_count': 0,
            'feature_assist_reactivation_count': 0,
            'memory_reactivation_count': 0,
            'overlap_disambiguation_count': 0,
            'overlap_disambiguation_helped_count': 0,
            'reactivating_state_count': 0,
            'predicted_only_to_tracked_count': 0,
            'predicted_candidate_reactivation_count': 0,
        }

    def _empty_group_bucket(self):
        bucket = {
            'frames': 0,
            'id_switch_frames': 0,
            'id_switch_events': 0,
            'unmatched_gt': 0,
            'unmatched_pred': 0,
            'rgb_dominant_association_count': 0,
            'ir_dominant_association_count': 0,
            'balanced_association_count': 0,
            'low_confidence_motion_fallback_count': 0,
            'modality_helped_reactivation_count': 0,
        }
        bucket.update(self._empty_runtime_frame_summary())
        return bucket
