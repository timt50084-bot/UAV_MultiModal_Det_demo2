from collections import defaultdict
from copy import deepcopy

import numpy as np
import torch

from src.metrics.task_specific_metrics import bbox_area
from src.model.bbox_utils import batch_prob_iou


def _safe_int(value, default=None):
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value, default=0.0):
    if value is None:
        return None if default is None else float(default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None if default is None else float(default)


def _safe_bool(value, default=False):
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {'1', 'true', 'yes', 'y', 'on'}
    return bool(value)


def _normalize_obb(box):
    if box is None:
        return None
    array = np.asarray(box, dtype=np.float32).reshape(-1)
    if array.size < 5:
        return None
    return array[:5].astype(np.float32)


def _normalize_track_object(obj, kind='pred'):
    if obj is None:
        return None
    track_id = obj.get('track_id', obj.get('id', obj.get('target_id', obj.get('instance_id'))))
    obb = _normalize_obb(obj.get('obb', obj.get('bbox', obj.get('box'))))
    if obb is None:
        return None
    default_score = 1.0 if kind == 'gt' else 0.0
    return {
        'track_id': _safe_int(track_id),
        'class_id': _safe_int(obj.get('class_id', obj.get('class', obj.get('cls_id', -1))), default=-1),
        'score': _safe_float(obj.get('score', default_score), default=default_score),
        'obb': obb,
        'state': obj.get('state', 'gt' if kind == 'gt' else 'tracked'),
        'size_group': obj.get('size_group'),
        'occlusion': obj.get('occlusion', obj.get('occlusion_level')),
        'time_of_day': obj.get('time_of_day'),
        'weather': obj.get('weather'),
        'rgb_reliability': _safe_float(obj.get('rgb_reliability'), default=None),
        'ir_reliability': _safe_float(obj.get('ir_reliability'), default=None),
        'fused_reliability': _safe_float(obj.get('fused_reliability'), default=None),
        'aggregated_rgb_reliability': _safe_float(obj.get('aggregated_rgb_reliability'), default=None),
        'aggregated_ir_reliability': _safe_float(obj.get('aggregated_ir_reliability'), default=None),
        'aggregated_fused_reliability': _safe_float(obj.get('aggregated_fused_reliability'), default=None),
        'association_mode': obj.get('association_mode'),
        'low_confidence_motion_fallback': _safe_bool(obj.get('low_confidence_motion_fallback', False), default=False),
        'modality_helped_reactivation': _safe_bool(obj.get('modality_helped_reactivation', False), default=False),
        'scene_adapted': _safe_bool(obj.get('scene_adapted', False), default=False),
        'dynamic_weights': deepcopy(obj.get('dynamic_weights')) if isinstance(obj.get('dynamic_weights'), dict) else None,
        'has_feature_assist': _safe_bool(obj.get('has_feature_assist', False), default=False),
        'has_aggregated_feature_assist': _safe_bool(obj.get('has_aggregated_feature_assist', False), default=False),
        'reactivation_source': obj.get('reactivation_source'),
        'memory_reactivation': _safe_bool(obj.get('memory_reactivation', False), default=False),
        'feature_assist_reactivation': _safe_bool(obj.get('feature_assist_reactivation', False), default=False),
        'predicted_candidate_reactivation': _safe_bool(obj.get('predicted_candidate_reactivation', False), default=False),
        'feature_assist_similarity': _safe_float(obj.get('feature_assist_similarity'), default=None),
        'memory_similarity': _safe_float(obj.get('memory_similarity'), default=None),
        'overlap_disambiguated': _safe_bool(obj.get('overlap_disambiguated', False), default=False),
        'overlap_disambiguation_helped': _safe_bool(obj.get('overlap_disambiguation_helped', False), default=False),
        'predicted_only_to_tracked': _safe_bool(obj.get('predicted_only_to_tracked', False), default=False),
        'state_transition': obj.get('state_transition'),
        'refinement_source': obj.get('refinement_source'),
        'rescued_detection': _safe_bool(obj.get('rescued_detection', False), default=False),
        'rescued_small_object': _safe_bool(obj.get('rescued_small_object', False), default=False),
        'predicted_candidate': _safe_bool(obj.get('predicted_candidate', False), default=False),
        'refinement_helped_reactivation': _safe_bool(obj.get('refinement_helped_reactivation', False), default=False),
        'support_track_id': _safe_int(obj.get('support_track_id'), default=None),
    }


def normalize_tracking_sequence(sequence, kind='pred', default_sequence_id='default'):
    if sequence is None:
        return None
    if isinstance(sequence, list):
        sequence = {'sequence_id': default_sequence_id, 'frames': sequence}

    sequence_id = str(sequence.get('sequence_id', default_sequence_id))
    frame_items = sequence.get('frames', sequence.get('results', []))
    frames = []
    for index, frame in enumerate(frame_items):
        frame_index = _safe_int(frame.get('frame_index', index), default=index)
        results = frame.get('results', frame.get('objects', frame.get('detections', [])))
        normalized_results = []
        for item in results:
            normalized = _normalize_track_object(item, kind=kind)
            if normalized is not None:
                normalized_results.append(normalized)

        metadata = dict(frame.get('metadata', {}))
        for key in ('time_of_day', 'weather', 'occlusion', 'occlusion_level', 'size_group'):
            if key in frame and key not in metadata:
                metadata[key] = frame.get(key)

        frames.append(
            {
                'sequence_id': sequence_id,
                'frame_index': frame_index,
                'image_id': frame.get('image_id', f'{sequence_id}_{frame_index:06d}'),
                'results': normalized_results,
                'metadata': metadata,
                'refinement_summary': deepcopy(frame.get('refinement_summary', {})),
                'advanced_summary': deepcopy(frame.get('advanced_summary', {})),
            }
        )

    frames.sort(key=lambda item: item['frame_index'])
    return {'sequence_id': sequence_id, 'frames': frames}


def normalize_tracking_sequences(data, kind='pred', default_sequence_id='default'):
    if data is None:
        return {}
    if isinstance(data, list):
        if not data:
            return {}
        if isinstance(data[0], dict) and ('frame_index' in data[0] or 'results' in data[0] or 'objects' in data[0]):
            sequence = normalize_tracking_sequence(data, kind=kind, default_sequence_id=default_sequence_id)
            return {sequence['sequence_id']: sequence}
        sequences = {}
        for index, item in enumerate(data):
            normalized = normalize_tracking_sequence(item, kind=kind, default_sequence_id=f'{default_sequence_id}_{index}')
            if normalized is not None:
                sequences[normalized['sequence_id']] = normalized
        return sequences
    if isinstance(data, dict):
        if 'frames' in data or 'results' in data:
            sequence = normalize_tracking_sequence(data, kind=kind, default_sequence_id=default_sequence_id)
            return {sequence['sequence_id']: sequence}
        sequences = {}
        for key, value in data.items():
            normalized = normalize_tracking_sequence(value, kind=kind, default_sequence_id=str(key))
            if normalized is not None:
                sequences[normalized['sequence_id']] = normalized
        return sequences
    raise TypeError('Tracking sequence data must be a list or dict.')


def _pairwise_hbb_iou(gt_boxes, pred_boxes):
    if gt_boxes.size == 0 or pred_boxes.size == 0:
        return np.zeros((gt_boxes.shape[0], pred_boxes.shape[0]), dtype=np.float32)

    gt_x1 = gt_boxes[:, 0] - gt_boxes[:, 2] * 0.5
    gt_y1 = gt_boxes[:, 1] - gt_boxes[:, 3] * 0.5
    gt_x2 = gt_boxes[:, 0] + gt_boxes[:, 2] * 0.5
    gt_y2 = gt_boxes[:, 1] + gt_boxes[:, 3] * 0.5
    pred_x1 = pred_boxes[:, 0] - pred_boxes[:, 2] * 0.5
    pred_y1 = pred_boxes[:, 1] - pred_boxes[:, 3] * 0.5
    pred_x2 = pred_boxes[:, 0] + pred_boxes[:, 2] * 0.5
    pred_y2 = pred_boxes[:, 1] + pred_boxes[:, 3] * 0.5

    inter_x1 = np.maximum(gt_x1[:, None], pred_x1[None, :])
    inter_y1 = np.maximum(gt_y1[:, None], pred_y1[None, :])
    inter_x2 = np.minimum(gt_x2[:, None], pred_x2[None, :])
    inter_y2 = np.minimum(gt_y2[:, None], pred_y2[None, :])
    inter_w = np.maximum(inter_x2 - inter_x1, 0.0)
    inter_h = np.maximum(inter_y2 - inter_y1, 0.0)
    inter = inter_w * inter_h

    area_gt = np.maximum(gt_boxes[:, 2], 0.0) * np.maximum(gt_boxes[:, 3], 0.0)
    area_pred = np.maximum(pred_boxes[:, 2], 0.0) * np.maximum(pred_boxes[:, 3], 0.0)
    union = area_gt[:, None] + area_pred[None, :] - inter
    return np.where(union > 0.0, inter / union, 0.0).astype(np.float32)


def pairwise_tracking_iou(gt_boxes, pred_boxes, use_obb_iou=True):
    gt_boxes = np.asarray(gt_boxes, dtype=np.float32)
    pred_boxes = np.asarray(pred_boxes, dtype=np.float32)
    if gt_boxes.size == 0 or pred_boxes.size == 0:
        return np.zeros((gt_boxes.shape[0], pred_boxes.shape[0]), dtype=np.float32)
    if not use_obb_iou:
        return _pairwise_hbb_iou(gt_boxes, pred_boxes)

    merged = torch.cat(
        [
            torch.as_tensor(gt_boxes, dtype=torch.float32),
            torch.as_tensor(pred_boxes, dtype=torch.float32),
        ],
        dim=0,
    )
    iou_matrix = batch_prob_iou(merged).cpu().numpy().astype(np.float32)
    split = gt_boxes.shape[0]
    return iou_matrix[:split, split:]


def _solve_max_weight_assignment(weights):
    weights = np.asarray(weights, dtype=np.float64)
    if weights.size == 0:
        return []

    original_rows, original_cols = weights.shape
    transposed = False
    matrix = weights
    if original_rows > original_cols:
        matrix = matrix.T
        transposed = True

    rows, cols = matrix.shape
    max_weight = float(matrix.max()) if matrix.size else 0.0
    cost = max_weight - matrix

    u = np.zeros(rows + 1, dtype=np.float64)
    v = np.zeros(cols + 1, dtype=np.float64)
    p = np.zeros(cols + 1, dtype=np.int64)
    way = np.zeros(cols + 1, dtype=np.int64)

    for row in range(1, rows + 1):
        p[0] = row
        j0 = 0
        minv = np.full(cols + 1, np.inf, dtype=np.float64)
        used = np.zeros(cols + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = np.inf
            j1 = 0
            for col in range(1, cols + 1):
                if used[col]:
                    continue
                cur = cost[i0 - 1, col - 1] - u[i0] - v[col]
                if cur < minv[col]:
                    minv[col] = cur
                    way[col] = j0
                if minv[col] < delta:
                    delta = minv[col]
                    j1 = col
            for col in range(cols + 1):
                if used[col]:
                    u[p[col]] += delta
                    v[col] -= delta
                else:
                    minv[col] -= delta
            j0 = j1
            if p[j0] == 0:
                break

        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    assignments = []
    for col in range(1, cols + 1):
        if p[col] == 0:
            continue
        row = p[col] - 1
        col_idx = col - 1
        assignments.append((col_idx, row) if transposed else (row, col_idx))
    return assignments


def match_tracking_frame(gt_objects, pred_objects, matching_cfg=None):
    matching_cfg = dict(matching_cfg or {})
    iou_threshold = float(matching_cfg.get('iou_threshold', 0.5))
    use_obb_iou = bool(matching_cfg.get('use_obb_iou', True))
    class_aware = bool(matching_cfg.get('class_aware', True))

    if not gt_objects or not pred_objects:
        return [], list(range(len(gt_objects))), list(range(len(pred_objects))), np.zeros((len(gt_objects), len(pred_objects)), dtype=np.float32)

    gt_boxes = np.stack([item['obb'] for item in gt_objects], axis=0)
    pred_boxes = np.stack([item['obb'] for item in pred_objects], axis=0)
    iou_matrix = pairwise_tracking_iou(gt_boxes, pred_boxes, use_obb_iou=use_obb_iou)

    if class_aware:
        gt_classes = np.asarray([item['class_id'] for item in gt_objects], dtype=np.int64)
        pred_classes = np.asarray([item['class_id'] for item in pred_objects], dtype=np.int64)
        class_mask = gt_classes[:, None] == pred_classes[None, :]
    else:
        class_mask = np.ones_like(iou_matrix, dtype=bool)

    weights = np.where(class_mask & (iou_matrix >= iou_threshold), iou_matrix, 0.0)
    raw_matches = _solve_max_weight_assignment(weights)

    matches = []
    matched_gt = set()
    matched_pred = set()
    for gt_idx, pred_idx in raw_matches:
        if gt_idx in matched_gt or pred_idx in matched_pred:
            continue
        if weights[gt_idx, pred_idx] < iou_threshold:
            continue
        matches.append((gt_idx, pred_idx, float(iou_matrix[gt_idx, pred_idx])))
        matched_gt.add(gt_idx)
        matched_pred.add(pred_idx)

    unmatched_gt = [index for index in range(len(gt_objects)) if index not in matched_gt]
    unmatched_pred = [index for index in range(len(pred_objects)) if index not in matched_pred]
    return matches, unmatched_gt, unmatched_pred, iou_matrix


def _summarize_track_flags(track_stats):
    mostly_tracked = 0
    mostly_lost = 0
    fragmentations = 0
    recovered_tracks = 0
    recovery_opportunities = 0

    for stats in track_stats.values():
        total_frames = max(int(stats['total_frames']), 1)
        matched_ratio = float(stats['matched_frames']) / float(total_frames)
        stats['matched_ratio'] = matched_ratio

        matched_flags = list(stats['matched_flags'])
        segments = 0
        previous = False
        gap_after_match = False
        reactivated = False
        for flag in matched_flags:
            if flag and not previous:
                segments += 1
                if gap_after_match:
                    reactivated = True
            if previous and not flag:
                gap_after_match = True
            previous = flag

        fragmentation = max(segments - 1, 0)
        stats['matched_segments'] = segments
        stats['fragmentations'] = fragmentation
        stats['reactivated'] = reactivated
        if fragmentation > 0:
            recovery_opportunities += 1
        if reactivated:
            recovered_tracks += 1

        if matched_ratio >= 0.8:
            mostly_tracked += 1
        if matched_ratio <= 0.2:
            mostly_lost += 1
        fragmentations += fragmentation

    return {
        'MostlyTracked': int(mostly_tracked),
        'MostlyLost': int(mostly_lost),
        'Fragmentations': int(fragmentations),
        'RecoveredTracks': int(recovered_tracks),
        'RecoveryOpportunities': int(recovery_opportunities),
    }


def evaluate_mot_sequence(pred_tracks, gt_tracks, config=None):
    config = dict(config or {})
    matching_cfg = dict(config.get('matching', {}))

    pred_sequence = normalize_tracking_sequence(pred_tracks, kind='pred', default_sequence_id='pred_sequence')
    gt_sequence = normalize_tracking_sequence(gt_tracks, kind='gt', default_sequence_id=pred_sequence['sequence_id'])

    if gt_sequence is None or not gt_sequence['frames']:
        return {'available': False, 'reason': 'missing_tracking_gt', 'sequence_id': pred_sequence['sequence_id'], 'metrics': None, 'details': None}

    gt_track_ids = [item['track_id'] for frame in gt_sequence['frames'] for item in frame['results'] if item.get('track_id') is not None]
    if not gt_track_ids:
        return {'available': False, 'reason': 'missing_gt_track_ids', 'sequence_id': gt_sequence['sequence_id'], 'metrics': None, 'details': None}

    pred_by_frame = {int(frame['frame_index']): frame for frame in pred_sequence['frames']}
    gt_by_frame = {int(frame['frame_index']): frame for frame in gt_sequence['frames']}
    all_frame_indices = sorted(set(pred_by_frame.keys()) | set(gt_by_frame.keys()))

    gt_track_stats = defaultdict(lambda: {'class_id': None, 'total_frames': 0, 'matched_frames': 0, 'matched_flags': [], 'areas': [], 'id_switches': 0, 'matched_pred_ids': [], 'frame_indices': [], 'size_group': None})
    pred_track_stats = defaultdict(lambda: {'class_id': None, 'total_frames': 0, 'matched_frames': 0, 'matched_gt_ids': set()})
    overlap_counts = defaultdict(lambda: defaultdict(int))
    last_pred_by_gt = {}
    id_switches = 0
    false_positives = 0
    false_negatives = 0
    total_gt = 0
    total_pred = 0
    matched_total = 0
    frame_summaries = []

    for frame_index in all_frame_indices:
        gt_frame = gt_by_frame.get(frame_index, {'results': [], 'metadata': {}, 'image_id': f'gt_{frame_index:06d}'})
        pred_frame = pred_by_frame.get(frame_index, {'results': [], 'metadata': {}, 'image_id': f'pred_{frame_index:06d}'})
        gt_objects = [item for item in gt_frame['results'] if item.get('track_id') is not None]
        pred_objects = [item for item in pred_frame['results'] if item.get('track_id') is not None]

        total_gt += len(gt_objects)
        total_pred += len(pred_objects)

        for gt in gt_objects:
            gt_id = int(gt['track_id'])
            stats = gt_track_stats[gt_id]
            stats['class_id'] = gt['class_id']
            stats['total_frames'] += 1
            stats['areas'].append(float(bbox_area(gt['obb'])))
            stats['frame_indices'].append(frame_index)
            stats['size_group'] = gt.get('size_group', stats.get('size_group'))

        for pred in pred_objects:
            pred_id = int(pred['track_id'])
            pred_stats = pred_track_stats[pred_id]
            pred_stats['class_id'] = pred['class_id']
            pred_stats['total_frames'] += 1

        matches, unmatched_gt, unmatched_pred, _ = match_tracking_frame(gt_objects, pred_objects, matching_cfg=matching_cfg)
        false_negatives += len(unmatched_gt)
        false_positives += len(unmatched_pred)
        matched_total += len(matches)
        match_lookup = {gt_idx: (pred_idx, iou) for gt_idx, pred_idx, iou in matches}
        id_switch_events = []
        matched_records = []

        for gt_idx, gt in enumerate(gt_objects):
            gt_id = int(gt['track_id'])
            stats = gt_track_stats[gt_id]
            if gt_idx not in match_lookup:
                stats['matched_flags'].append(False)
                stats['matched_pred_ids'].append(None)
                continue

            pred_idx, iou = match_lookup[gt_idx]
            pred = pred_objects[pred_idx]
            pred_id = int(pred['track_id'])
            stats['matched_frames'] += 1
            stats['matched_flags'].append(True)
            stats['matched_pred_ids'].append(pred_id)
            pred_track_stats[pred_id]['matched_frames'] += 1
            pred_track_stats[pred_id]['matched_gt_ids'].add(gt_id)
            overlap_counts[gt_id][pred_id] += 1

            previous_pred_id = last_pred_by_gt.get(gt_id)
            switched = previous_pred_id is not None and previous_pred_id != pred_id
            if switched:
                id_switches += 1
                stats['id_switches'] += 1
                id_switch_events.append({'gt_track_id': gt_id, 'from_pred_track_id': previous_pred_id, 'to_pred_track_id': pred_id})
            last_pred_by_gt[gt_id] = pred_id
            matched_records.append(
                {
                    'gt_track_id': gt_id,
                    'pred_track_id': pred_id,
                    'gt_class_id': gt['class_id'],
                    'pred_class_id': pred['class_id'],
                    'iou': float(iou),
                    'id_switch': switched,
                    'association_mode': pred.get('association_mode'),
                    'low_confidence_motion_fallback': bool(pred.get('low_confidence_motion_fallback', False)),
                    'modality_helped_reactivation': bool(pred.get('modality_helped_reactivation', False)),
                    'scene_adapted': bool(pred.get('scene_adapted', False)),
                    'rgb_reliability': pred.get('rgb_reliability'),
                    'ir_reliability': pred.get('ir_reliability'),
                    'fused_reliability': pred.get('fused_reliability'),
                    'aggregated_rgb_reliability': pred.get('aggregated_rgb_reliability'),
                    'aggregated_ir_reliability': pred.get('aggregated_ir_reliability'),
                    'aggregated_fused_reliability': pred.get('aggregated_fused_reliability'),
                    'dynamic_weights': deepcopy(pred.get('dynamic_weights')) if isinstance(pred.get('dynamic_weights'), dict) else None,
                    'reactivation_source': pred.get('reactivation_source'),
                    'memory_reactivation': bool(pred.get('memory_reactivation', False)),
                    'feature_assist_reactivation': bool(pred.get('feature_assist_reactivation', False)),
                    'predicted_candidate_reactivation': bool(pred.get('predicted_candidate_reactivation', False)),
                    'feature_assist_similarity': pred.get('feature_assist_similarity'),
                    'memory_similarity': pred.get('memory_similarity'),
                    'overlap_disambiguated': bool(pred.get('overlap_disambiguated', False)),
                    'overlap_disambiguation_helped': bool(pred.get('overlap_disambiguation_helped', False)),
                    'predicted_only_to_tracked': bool(pred.get('predicted_only_to_tracked', False)),
                    'state_transition': pred.get('state_transition'),
                    'refinement_source': pred.get('refinement_source'),
                    'rescued_detection': bool(pred.get('rescued_detection', False)),
                    'rescued_small_object': bool(pred.get('rescued_small_object', False)),
                    'predicted_candidate': bool(pred.get('predicted_candidate', False)),
                    'refinement_helped_reactivation': bool(pred.get('refinement_helped_reactivation', False)),
                    'support_track_id': pred.get('support_track_id'),
                }
            )

        frame_summaries.append(
            {
                'sequence_id': gt_sequence['sequence_id'],
                'frame_index': frame_index,
                'image_id': gt_frame.get('image_id', pred_frame.get('image_id')),
                'metadata': deepcopy(gt_frame.get('metadata') or pred_frame.get('metadata') or {}),
                'gt_count': len(gt_objects),
                'pred_count': len(pred_objects),
                'matches': matched_records,
                'pred_results': deepcopy(pred_objects),
                'refinement_summary': deepcopy(pred_frame.get('refinement_summary', {})),
                'advanced_summary': deepcopy(pred_frame.get('advanced_summary', {})),
                'unmatched_gt_track_ids': [gt_objects[index]['track_id'] for index in unmatched_gt],
                'unmatched_pred_track_ids': [pred_objects[index]['track_id'] for index in unmatched_pred],
                'id_switch_events': id_switch_events,
            }
        )

    track_summary = _summarize_track_flags(gt_track_stats)
    total_gt_tracks = len(gt_track_stats)
    total_pred_tracks = len(pred_track_stats)
    matched_gt_tracks = sum(1 for stats in gt_track_stats.values() if stats['matched_frames'] > 0)
    matched_pred_tracks = sum(1 for stats in pred_track_stats.values() if stats['matched_frames'] > 0)

    gt_ids = sorted(overlap_counts.keys())
    pred_ids = sorted({pred_id for pred_map in overlap_counts.values() for pred_id in pred_map.keys()})
    if gt_ids and pred_ids:
        weight_matrix = np.zeros((len(gt_ids), len(pred_ids)), dtype=np.float32)
        for gt_index, gt_id in enumerate(gt_ids):
            for pred_index, pred_id in enumerate(pred_ids):
                weight_matrix[gt_index, pred_index] = float(overlap_counts[gt_id].get(pred_id, 0))
        assignment = _solve_max_weight_assignment(weight_matrix)
        idtp = sum(int(weight_matrix[gt_idx, pred_idx]) for gt_idx, pred_idx in assignment if gt_idx < len(gt_ids) and pred_idx < len(pred_ids))
    else:
        idtp = 0

    idfp = max(total_pred - idtp, 0)
    idfn = max(total_gt - idtp, 0)
    mota = None if total_gt == 0 else 1.0 - float(false_negatives + false_positives + id_switches) / float(total_gt)
    idf1_denom = (2 * idtp) + idfp + idfn
    idf1 = float(2 * idtp / idf1_denom) if idf1_denom > 0 else 0.0

    metrics = {
        'MOTA': float(mota) if mota is not None else None,
        'IDF1': float(idf1),
        'IDSwitches': int(id_switches),
        'MostlyTracked': int(track_summary['MostlyTracked']),
        'MostlyLost': int(track_summary['MostlyLost']),
        'Fragmentations': int(track_summary['Fragmentations']),
        'TrackRecall': float(matched_gt_tracks / max(total_gt_tracks, 1)),
        'TrackPrecision': float(matched_pred_tracks / max(total_pred_tracks, 1)),
        'num_frames': int(len(all_frame_indices)),
        'num_gt_tracks': int(total_gt_tracks),
        'num_pred_tracks': int(total_pred_tracks),
        'num_gt_detections': int(total_gt),
        'num_pred_detections': int(total_pred),
        'matches': int(matched_total),
        'false_positives': int(false_positives),
        'false_negatives': int(false_negatives),
        'IDTP': int(idtp),
        'IDFP': int(idfp),
        'IDFN': int(idfn),
        'RecoveredTracks': int(track_summary['RecoveredTracks']),
        'RecoveryOpportunities': int(track_summary['RecoveryOpportunities']),
    }

    details = {
        'frame_summaries': frame_summaries,
        'gt_track_stats': {
            str(track_id): {
                **deepcopy(stats),
                'matched_gt_id': int(track_id),
                'matched_ratio': float(stats.get('matched_ratio', 0.0)),
                'areas': [float(value) for value in stats.get('areas', [])],
            }
            for track_id, stats in gt_track_stats.items()
        },
        'pred_track_stats': {
            str(track_id): {
                **deepcopy(stats),
                'matched_gt_ids': sorted(int(value) for value in stats.get('matched_gt_ids', set())),
            }
            for track_id, stats in pred_track_stats.items()
        },
        'overlap_counts': {
            str(gt_id): {str(pred_id): int(count) for pred_id, count in pred_map.items()}
            for gt_id, pred_map in overlap_counts.items()
        },
    }
    return {'available': True, 'reason': None, 'sequence_id': gt_sequence['sequence_id'], 'metrics': metrics, 'details': details}


def evaluate_mot_dataset(sequence_results, gt_sequences, config=None):
    config = dict(config or {})
    pred_sequences = normalize_tracking_sequences(sequence_results, kind='pred', default_sequence_id='pred')
    gt_sequence_map = normalize_tracking_sequences(gt_sequences, kind='gt', default_sequence_id='gt')

    if not gt_sequence_map:
        return {'available': False, 'reason': 'missing_tracking_gt', 'metrics': None, 'per_sequence': {}}
    if not pred_sequences:
        return {'available': False, 'reason': 'missing_tracking_predictions', 'metrics': None, 'per_sequence': {}}

    common_ids = [sequence_id for sequence_id in pred_sequences.keys() if sequence_id in gt_sequence_map]
    if not common_ids:
        return {'available': False, 'reason': 'no_common_tracking_sequences', 'metrics': None, 'per_sequence': {}}

    per_sequence = {}
    aggregate = defaultdict(float)
    total_gt_tracks = 0
    total_pred_tracks = 0
    evaluated_sequences = 0

    for sequence_id in common_ids:
        result = evaluate_mot_sequence(pred_sequences[sequence_id], gt_sequence_map[sequence_id], config=config)
        if not result['available']:
            continue
        evaluated_sequences += 1
        per_sequence[sequence_id] = result
        metrics = result['metrics']
        for key in ('false_positives', 'false_negatives', 'IDSwitches', 'num_gt_detections', 'num_pred_detections', 'IDTP', 'IDFP', 'IDFN', 'Fragmentations', 'MostlyTracked', 'MostlyLost', 'RecoveredTracks', 'RecoveryOpportunities'):
            aggregate[key] += float(metrics.get(key, 0))
        total_gt_tracks += int(metrics.get('num_gt_tracks', 0))
        total_pred_tracks += int(metrics.get('num_pred_tracks', 0))

    if evaluated_sequences == 0:
        return {'available': False, 'reason': 'no_common_tracking_sequences', 'metrics': None, 'per_sequence': {}}

    total_gt = max(int(aggregate['num_gt_detections']), 0)
    total_pred = max(int(aggregate['num_pred_detections']), 0)
    total_idtp = max(int(aggregate['IDTP']), 0)
    total_idfp = max(int(aggregate['IDFP']), 0)
    total_idfn = max(int(aggregate['IDFN']), 0)
    mota = None if total_gt == 0 else 1.0 - float(aggregate['false_negatives'] + aggregate['false_positives'] + aggregate['IDSwitches']) / float(total_gt)
    idf1_denom = (2 * total_idtp) + total_idfp + total_idfn
    idf1 = float(2 * total_idtp / idf1_denom) if idf1_denom > 0 else 0.0

    matched_gt_tracks = sum(1 for sequence_result in per_sequence.values() for stats in sequence_result['details']['gt_track_stats'].values() if int(stats.get('matched_frames', 0)) > 0)
    matched_pred_tracks = sum(1 for sequence_result in per_sequence.values() for stats in sequence_result['details']['pred_track_stats'].values() if int(stats.get('matched_frames', 0)) > 0)

    metrics = {
        'MOTA': float(mota) if mota is not None else None,
        'IDF1': float(idf1),
        'IDSwitches': int(aggregate['IDSwitches']),
        'MostlyTracked': int(aggregate['MostlyTracked']),
        'MostlyLost': int(aggregate['MostlyLost']),
        'Fragmentations': int(aggregate['Fragmentations']),
        'TrackRecall': float(matched_gt_tracks / max(total_gt_tracks, 1)),
        'TrackPrecision': float(matched_pred_tracks / max(total_pred_tracks, 1)),
        'RecoveredTracks': int(aggregate['RecoveredTracks']),
        'RecoveryOpportunities': int(aggregate['RecoveryOpportunities']),
        'num_sequences': int(evaluated_sequences),
        'num_gt_tracks': int(total_gt_tracks),
        'num_pred_tracks': int(total_pred_tracks),
        'num_gt_detections': int(total_gt),
        'num_pred_detections': int(total_pred),
        'IDTP': int(total_idtp),
        'IDFP': int(total_idfp),
        'IDFN': int(total_idfn),
        'false_positives': int(aggregate['false_positives']),
        'false_negatives': int(aggregate['false_negatives']),
    }
    return {'available': True, 'reason': None, 'metrics': metrics, 'per_sequence': per_sequence}


