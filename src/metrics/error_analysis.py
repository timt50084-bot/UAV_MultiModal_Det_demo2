from collections import defaultdict
from copy import deepcopy

import numpy as np

from src.metrics.obb_metrics import polygon_iou
from src.metrics.task_metrics import normalize_eval_metrics_cfg
from src.metrics.task_specific_metrics import (
    angle_distance,
    bbox_area,
    is_small_bbox,
    resolve_small_area_threshold,
)
from src.utils.analysis_io import export_error_analysis


class ErrorAnalyzer:
    def __init__(self, cfg=None, class_names=None, match_iou_fn=None):
        self.eval_cfg = normalize_eval_metrics_cfg(cfg)
        self.cfg = self.eval_cfg['error_analysis']
        self.class_names = list(class_names or [])
        self.match_iou_fn = match_iou_fn

    def _class_name(self, class_id):
        class_id = int(class_id)
        if 0 <= class_id < len(self.class_names):
            return str(self.class_names[class_id])
        return f'class_{class_id}'

    def _default_match_iou_fn(self):
        poly_cache = {}
        return lambda pred_box, gt_box: polygon_iou(pred_box, gt_box, poly_cache)

    def _area_bucket(self, obb):
        linear_size = float(np.sqrt(max(bbox_area(obb), 0.0)))
        bins = self.cfg['buckets'].get('area_bins', [16, 32, 96])
        if linear_size < bins[0]:
            return 'tiny'
        if linear_size < bins[1]:
            return 'small'
        if linear_size < bins[2]:
            return 'medium'
        return 'large'

    def _aspect_ratio_bucket(self, obb):
        w = max(float(obb[2]), 1e-6)
        h = max(float(obb[3]), 1e-6)
        ratio = max(w, h) / min(w, h)
        bins = self.cfg['buckets'].get('aspect_ratio_bins', [1.5, 3.0])
        if ratio <= bins[0]:
            return 'near_square'
        if ratio <= bins[1]:
            return 'elongated'
        return 'highly_elongated'

    def _is_dense_scene(self, metadata, gt_count):
        if isinstance(metadata, dict):
            occlusion_value = metadata.get('occlusion_level', metadata.get('occlusion'))
            if isinstance(occlusion_value, str) and occlusion_value.lower() in {'dense', 'heavy', 'high'}:
                return True
        return gt_count >= int(self.cfg['buckets'].get('dense_scene_gt_threshold', 5))

    def _is_low_visibility(self, metadata):
        if not isinstance(metadata, dict):
            return False
        time_of_day = str(metadata.get('time_of_day', '')).lower()
        weather = str(metadata.get('weather', '')).lower()
        visibility = str(metadata.get('visibility', '')).lower()
        return time_of_day in {'night', 'evening'} or weather in {'fog', 'rain', 'low_light'} or visibility in {'low', 'poor'}

    def _group_by_image(self, entries):
        grouped = defaultdict(list)
        for entry in entries or []:
            grouped[entry['image_id']].append(entry)
        return grouped

    def _match_single_image(self, image_preds, image_gts, match_iou_fn, iou_threshold):
        sorted_preds = sorted(image_preds, key=lambda item: item.get('score', 0.0), reverse=True)
        gt_matched = [False] * len(image_gts)
        matched_pairs = []
        false_positives = []

        for pred in sorted_preds:
            best_iou = 0.0
            best_gt_idx = -1
            for gt_idx, gt in enumerate(image_gts):
                if gt_matched[gt_idx] or int(gt['class']) != int(pred['class']):
                    continue
                iou = match_iou_fn(pred['bbox'], gt['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            if best_iou >= iou_threshold and best_gt_idx >= 0:
                gt_matched[best_gt_idx] = True
                matched_pairs.append({
                    'pred': pred,
                    'gt': image_gts[best_gt_idx],
                    'iou': float(best_iou),
                    'angle_error': float(angle_distance(pred['bbox'][4], image_gts[best_gt_idx]['bbox'][4])),
                })
            else:
                false_positives.append(pred)

        false_negatives = [gt for gt_idx, gt in enumerate(image_gts) if not gt_matched[gt_idx]]
        return {
            'matched_pairs': matched_pairs,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
        }

    def _build_confusion_summary(self, image_preds, image_gts, false_positives, false_negatives, match_iou_fn, iou_threshold):
        confusion_summary = defaultdict(int)
        if not self.cfg['confusion'].get('enabled', True):
            return confusion_summary

        for pred in false_positives:
            best_iou = 0.0
            best_gt = None
            for gt in false_negatives:
                iou = match_iou_fn(pred['bbox'], gt['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_gt = gt

            if best_gt is not None and best_iou >= iou_threshold:
                key = f"{self._class_name(best_gt['class'])}->{self._class_name(pred['class'])}"
            else:
                key = f"background_fp->{self._class_name(pred['class'])}"
            confusion_summary[key] += 1

        return confusion_summary

    def _build_per_image_record(self, image_id, metadata, image_preds, image_gts, match_result, confusion_summary):
        small_threshold = resolve_small_area_threshold(self.cfg['buckets'].get('small_object_area_threshold', 32))
        gt_count = len(image_gts)
        tp_count = len(match_result['matched_pairs'])
        fp_count = len(match_result['false_positives'])
        fn_count = len(match_result['false_negatives'])
        small_gt_count = sum(1 for gt in image_gts if is_small_bbox(gt['bbox'], small_threshold))
        small_fn_count = sum(1 for gt in match_result['false_negatives'] if is_small_bbox(gt['bbox'], small_threshold))
        small_miss = small_fn_count > 0
        dense_occlusion_miss = fn_count > 0 and self._is_dense_scene(metadata, gt_count)
        low_visibility_miss = fn_count > 0 and self._is_low_visibility(metadata)

        main_error_category = 'clean'
        if small_miss:
            main_error_category = 'small_miss'
        elif dense_occlusion_miss:
            main_error_category = 'dense_occlusion_miss'
        elif low_visibility_miss:
            main_error_category = 'low_visibility_miss'
        elif confusion_summary:
            main_error_category = 'class_confusion'
        elif fn_count > 0:
            main_error_category = 'false_negative'
        elif fp_count > 0:
            main_error_category = 'false_positive'

        metadata = metadata or {}
        return {
            'image_id': image_id,
            'sequence_id': metadata.get('sequence_id'),
            'frame_index': metadata.get('frame_index'),
            'gt_count': gt_count,
            'pred_count': len(image_preds),
            'tp_count': tp_count,
            'fp_count': fp_count,
            'fn_count': fn_count,
            'small_gt_count': small_gt_count,
            'small_fn_count': small_fn_count,
            'small_miss': bool(small_miss),
            'dense_occlusion_miss': bool(dense_occlusion_miss),
            'main_error_category': main_error_category,
            'time_of_day': metadata.get('time_of_day'),
            'weather': metadata.get('weather'),
            'occlusion_level': metadata.get('occlusion_level', metadata.get('occlusion')),
            'size_group': metadata.get('size_group'),
        }

    def _summarize_angle_errors(self, angle_errors):
        if not angle_errors:
            return {
                'count': 0,
                'mean': 0.0,
                'median': 0.0,
                'max': 0.0,
                'unit': 'radian',
                'description': 'Matched prediction vs ground-truth angle error with pi-periodicity.',
            }

        values = np.array(angle_errors, dtype=np.float32)
        return {
            'count': int(values.size),
            'mean': float(values.mean()),
            'median': float(np.median(values)),
            'max': float(values.max()),
            'unit': 'radian',
            'description': 'Matched prediction vs ground-truth angle error with pi-periodicity.',
        }

    def _build_modality_contribution_summary(self, baseline_records, rgb_drop_data=None, ir_drop_data=None):
        summary = {
            'enabled': bool(self.cfg['modality_contribution'].get('enabled', False)),
            'available': False,
            'rgb_dominant_success_count': 0,
            'ir_dominant_success_count': 0,
            'dual_synergy_success_count': 0,
        }
        if not summary['enabled']:
            return summary
        if not rgb_drop_data or not ir_drop_data:
            return summary

        rgb_records = self._compute_image_record_map(
            rgb_drop_data.get('preds', []),
            rgb_drop_data.get('gts', []),
            rgb_drop_data.get('image_metadata', {}),
        )
        ir_records = self._compute_image_record_map(
            ir_drop_data.get('preds', []),
            ir_drop_data.get('gts', []),
            ir_drop_data.get('image_metadata', {}),
        )

        for image_id, baseline_record in baseline_records.items():
            if baseline_record['gt_count'] <= 0:
                continue
            baseline_success = baseline_record['fn_count'] == 0 and baseline_record['tp_count'] > 0
            rgb_record = rgb_records.get(image_id)
            ir_record = ir_records.get(image_id)
            if not baseline_success or rgb_record is None or ir_record is None:
                continue

            rgb_success = rgb_record['fn_count'] == 0 and rgb_record['tp_count'] > 0
            ir_success = ir_record['fn_count'] == 0 and ir_record['tp_count'] > 0
            if not rgb_success and ir_success:
                summary['rgb_dominant_success_count'] += 1
            elif rgb_success and not ir_success:
                summary['ir_dominant_success_count'] += 1
            elif not rgb_success and not ir_success:
                summary['dual_synergy_success_count'] += 1

        summary['available'] = True
        return summary

    def _compute_image_record_map(self, preds, gts, image_metadata):
        grouped_preds = self._group_by_image(preds)
        grouped_gts = self._group_by_image(gts)
        image_ids = sorted(set(grouped_preds.keys()) | set(grouped_gts.keys()) | set((image_metadata or {}).keys()))
        match_iou_fn = self.match_iou_fn or self._default_match_iou_fn()
        iou_threshold = float(self.cfg.get('iou_threshold', 0.5))

        image_record_map = {}
        for image_id in image_ids:
            image_preds = grouped_preds.get(image_id, [])
            image_gts = grouped_gts.get(image_id, [])
            metadata = (image_metadata or {}).get(image_id)
            match_result = self._match_single_image(image_preds, image_gts, match_iou_fn, iou_threshold)
            confusion_summary = self._build_confusion_summary(
                image_preds,
                image_gts,
                match_result['false_positives'],
                match_result['false_negatives'],
                match_iou_fn,
                iou_threshold,
            )
            image_record_map[image_id] = self._build_per_image_record(
                image_id,
                metadata,
                image_preds,
                image_gts,
                match_result,
                confusion_summary,
            )
        return image_record_map

    def analyze(self, preds, gts, image_metadata=None, rgb_drop_data=None, ir_drop_data=None, grouped_metrics=None):
        del grouped_metrics
        image_metadata = image_metadata or {}
        grouped_preds = self._group_by_image(preds)
        grouped_gts = self._group_by_image(gts)
        image_ids = sorted(set(grouped_preds.keys()) | set(grouped_gts.keys()) | set(image_metadata.keys()))
        match_iou_fn = self.match_iou_fn or self._default_match_iou_fn()
        iou_threshold = float(self.cfg.get('iou_threshold', 0.5))
        small_threshold = resolve_small_area_threshold(self.cfg['buckets'].get('small_object_area_threshold', 32))

        per_image_records = []
        class_confusion_summary = defaultdict(int)
        area_bucket_summary = {
            'tp': defaultdict(int),
            'fn': defaultdict(int),
            'fp': defaultdict(int),
        }
        aspect_ratio_bucket_summary = {
            'tp': defaultdict(int),
            'fn': defaultdict(int),
            'fp': defaultdict(int),
        }
        angle_errors = []
        false_negative_breakdown = {
            'small_object_miss': 0,
            'dense_occlusion_miss': 0,
            'low_visibility_miss': 0,
        }
        false_positive_breakdown = {
            'background_like_fp': 0,
            'reflection_like_fp': 0,
            'class_confusion_fp': 0,
        }
        total_tp = 0
        total_fp = 0
        total_fn = 0
        small_tp = 0
        small_fn = 0

        for image_id in image_ids:
            image_preds = grouped_preds.get(image_id, [])
            image_gts = grouped_gts.get(image_id, [])
            metadata = image_metadata.get(image_id)
            match_result = self._match_single_image(image_preds, image_gts, match_iou_fn, iou_threshold)
            confusion_summary = self._build_confusion_summary(
                image_preds,
                image_gts,
                match_result['false_positives'],
                match_result['false_negatives'],
                match_iou_fn,
                iou_threshold,
            )
            for key, value in confusion_summary.items():
                class_confusion_summary[key] += value

            total_tp += len(match_result['matched_pairs'])
            total_fp += len(match_result['false_positives'])
            total_fn += len(match_result['false_negatives'])

            for pair in match_result['matched_pairs']:
                area_bucket_summary['tp'][self._area_bucket(pair['gt']['bbox'])] += 1
                aspect_ratio_bucket_summary['tp'][self._aspect_ratio_bucket(pair['gt']['bbox'])] += 1
                angle_errors.append(pair['angle_error'])
                if is_small_bbox(pair['gt']['bbox'], small_threshold):
                    small_tp += 1

            for gt in match_result['false_negatives']:
                area_bucket_summary['fn'][self._area_bucket(gt['bbox'])] += 1
                aspect_ratio_bucket_summary['fn'][self._aspect_ratio_bucket(gt['bbox'])] += 1
                if is_small_bbox(gt['bbox'], small_threshold):
                    small_fn += 1

            for pred in match_result['false_positives']:
                aspect_ratio_bucket_summary['fp'][self._aspect_ratio_bucket(pred['bbox'])] += 1
                area_bucket_summary['fp'][self._area_bucket(pred['bbox'])] += 1

            record = self._build_per_image_record(
                image_id,
                metadata,
                image_preds,
                image_gts,
                match_result,
                confusion_summary,
            )
            per_image_records.append(record)

            if record['small_miss']:
                false_negative_breakdown['small_object_miss'] += 1
            if record['dense_occlusion_miss']:
                false_negative_breakdown['dense_occlusion_miss'] += 1
            if record['main_error_category'] == 'low_visibility_miss':
                false_negative_breakdown['low_visibility_miss'] += 1

            false_positive_breakdown['class_confusion_fp'] += sum(
                value for key, value in confusion_summary.items() if not key.startswith('background_fp->')
            )
            false_positive_breakdown['background_like_fp'] += sum(
                value for key, value in confusion_summary.items() if key.startswith('background_fp->')
            )
            if isinstance(metadata, dict):
                if str(metadata.get('highlight_region', '')).lower() in {'true', '1', 'yes'} or str(metadata.get('hotspot', '')).lower() in {'true', '1', 'yes'}:
                    false_positive_breakdown['reflection_like_fp'] += record['fp_count']

        image_record_map = {record['image_id']: record for record in per_image_records}
        modality_contribution_summary = self._build_modality_contribution_summary(
            image_record_map,
            rgb_drop_data=rgb_drop_data,
            ir_drop_data=ir_drop_data,
        )

        summary = {
            'total_fp': int(total_fp),
            'total_fn': int(total_fn),
            'total_tp': int(total_tp),
            'small_fn': int(small_fn),
            'small_tp': int(small_tp),
            'class_confusion_summary': dict(sorted(class_confusion_summary.items())),
            'area_bucket_summary': {
                key: dict(sorted(bucket_counts.items()))
                for key, bucket_counts in area_bucket_summary.items()
            },
            'aspect_ratio_bucket_summary': {
                key: dict(sorted(bucket_counts.items()))
                for key, bucket_counts in aspect_ratio_bucket_summary.items()
            },
            'angle_error_summary': self._summarize_angle_errors(angle_errors),
            'false_negative_breakdown': false_negative_breakdown,
            'false_positive_breakdown': false_positive_breakdown,
            'modality_contribution_summary': modality_contribution_summary,
        }

        exported_files = {}
        if self.cfg.get('enabled', False) and (self.cfg.get('export_json', True) or self.cfg.get('export_csv', True)):
            exported_files = export_error_analysis(
                summary=summary,
                per_image_records=per_image_records if self.cfg.get('include_per_image', True) else [],
                output_dir=self.cfg.get('output_dir', 'outputs/error_analysis'),
                export_json=bool(self.cfg.get('export_json', True)),
                export_csv=bool(self.cfg.get('export_csv', True)),
                include_per_image=bool(self.cfg.get('include_per_image', True)),
            )

        return {
            'summary': deepcopy(summary),
            'per_image_records': deepcopy(per_image_records),
            'exported_files': exported_files,
        }
