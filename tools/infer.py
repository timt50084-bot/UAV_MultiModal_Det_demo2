import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from src.model.bbox_utils import non_max_suppression_obb, xywhr2xyxyxyxy
from src.model.builder import build_model
from src.model.output_adapter import flatten_predictions
from src.tracking import (
    TrackAwareRefiner,
    build_tracker_from_cfg,
    detections_to_results,
    maybe_extract_detection_appearance_features,
    maybe_extract_detection_feature_assist_features,
    maybe_extract_detection_reliability_features,
    normalize_tracking_cfg,
)
from src.utils.config import load_config
from src.utils.config_utils import apply_experiment_runtime_overrides
from src.utils.detection_cuda import resolve_detection_device
from src.utils.postprocess_tuning import (
    apply_classwise_thresholds,
    describe_classwise_thresholds,
    describe_tta_settings,
    normalize_infer_cfg,
)
from src.utils.result_merge import merge_obb_predictions
from src.utils.tta import apply_tta_transforms, build_tta_transforms, invert_tta_predictions

COLORS = [(0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255), (0, 0, 255)]
IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
SCENE_CONTEXT_KEYS = ('time_of_day', 'weather', 'visibility')


def letterbox(img, new_shape=(1024, 1024), color=(114, 114, 114), stride=32):
    shape = img.shape[:2]
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw, dh = np.mod(dw, stride) / 2, np.mod(dh, stride) / 2
    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    return cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color), r, (dw, dh)


def draw_obb(image, obb_boxes, scores, classes, class_names, track_ids=None, states=None):
    polygons = xywhr2xyxyxyxy(obb_boxes).cpu().numpy()
    for i, poly in enumerate(polygons):
        cls_id = int(classes[i].item())
        class_name = class_names[cls_id] if 0 <= cls_id < len(class_names) else str(cls_id)
        label = f"{class_name} {scores[i].item():.2f}"
        if track_ids is not None and i < len(track_ids) and track_ids[i] is not None:
            label = f"ID{track_ids[i]} {label}"
        if states is not None and i < len(states) and states[i] not in {None, 'tracked', 'detection'}:
            label = f"{label} {states[i]}"
        color_index = track_ids[i] if track_ids is not None and i < len(track_ids) and track_ids[i] is not None else cls_id
        color = COLORS[color_index % len(COLORS)]
        poly = np.int32(poly)
        cv2.polylines(image, [poly], isClosed=True, color=color, thickness=2, lineType=cv2.LINE_AA)
        pt1 = tuple(poly[0])
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
        cv2.rectangle(image, pt1, (pt1[0] + t_size[0], pt1[1] - t_size[1] - 3), color, -1, cv2.LINE_AA)
        cv2.putText(image, label, (pt1[0], pt1[1] - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return image


def draw_heatmap(original_img, feature_map, save_path):
    heatmap = torch.mean(feature_map, dim=1).squeeze(0) if len(feature_map.shape) == 4 else feature_map.squeeze(0)
    heatmap = np.maximum(heatmap.cpu().float().numpy(), 0)
    if np.max(heatmap) > 0:
        heatmap /= np.max(heatmap)
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    heatmap_color = cv2.resize(heatmap_color, (original_img.shape[1], original_img.shape[0]))
    cv2.imwrite(str(save_path), cv2.addWeighted(original_img, 0.6, heatmap_color, 0.4, 0))
    print(f'Heatmap saved to {save_path}')


def to_tensor(image_bgr, swap_rb=False):
    image = image_bgr[..., ::-1] if swap_rb else image_bgr
    return torch.from_numpy(image.transpose(2, 0, 1).copy()).float().unsqueeze(0) / 255.0


def forward_flat_predictions(model, imgs_rgb, imgs_ir, prev_rgb, prev_ir, return_tracking_features=False):
    with torch.no_grad():
        if return_tracking_features:
            outputs, feat_rgb, feat_ir, tracking_feature_payload = model(
                imgs_rgb,
                imgs_ir,
                prev_rgb=prev_rgb,
                prev_ir=prev_ir,
                return_tracking_features=True,
            )
        else:
            outputs, feat_rgb, feat_ir = model(imgs_rgb, imgs_ir, prev_rgb=prev_rgb, prev_ir=prev_ir)
            tracking_feature_payload = None
        flat_preds, _ = flatten_predictions(outputs)
        flat_preds = torch.nan_to_num(flat_preds, nan=0.0, posinf=1e4, neginf=-1e4)
        flat_preds[..., :5] = flat_preds[..., :5].clamp(-1e4, 1e4)
        if flat_preds.shape[1] > 3000:
            _, idx = flat_preds[..., 5:].amax(dim=-1).topk(3000, dim=-1)
            flat_preds = flat_preds.gather(1, idx.unsqueeze(-1).expand(-1, -1, flat_preds.shape[-1]))
        flat_preds[..., 5:] = flat_preds[..., 5:].sigmoid()
    return flat_preds.float(), feat_rgb, feat_ir, tracking_feature_payload


def run_single_prediction(model, infer_cfg, class_names, imgs_rgb, imgs_ir, prev_rgb, prev_ir, tracking_cfg=None):
    return_tracking_features = bool(
        tracking_cfg
        and tracking_cfg.get('enabled', False)
        and (
            (tracking_cfg.get('appearance', {}).get('enabled', False) and tracking_cfg.get('association', {}).get('use_appearance', False))
            or (tracking_cfg.get('modality', {}).get('enabled', False) and tracking_cfg.get('association', {}).get('use_modality_awareness', False))
            or tracking_cfg.get('feature_assist', {}).get('enabled', False)
        )
    )
    flat_preds, feat_rgb, feat_ir, tracking_feature_payload = forward_flat_predictions(
        model,
        imgs_rgb,
        imgs_ir,
        prev_rgb,
        prev_ir,
        return_tracking_features=return_tracking_features,
    )
    refinement_cfg = tracking_cfg.get('refinement', {}) if isinstance(tracking_cfg, dict) else {}
    effective_conf_threshold = infer_cfg['conf_threshold']
    if tracking_cfg and tracking_cfg.get('enabled', False) and refinement_cfg.get('enabled', False) and refinement_cfg.get('rescue_low_score', False):
        effective_conf_threshold = min(float(infer_cfg['conf_threshold']), float(refinement_cfg.get('rescue_score_threshold', infer_cfg['conf_threshold'])))
    preds = non_max_suppression_obb(
        flat_preds,
        conf_thres=effective_conf_threshold,
        iou_thres=infer_cfg['iou_threshold'],
        max_det=infer_cfg['merge']['max_det'],
    )[0]
    preds = apply_classwise_thresholds(
        preds,
        class_names=class_names,
        global_conf_threshold=effective_conf_threshold,
        classwise_conf_thresholds=infer_cfg.get('classwise_conf_thresholds', {}),
    )
    return preds, feat_rgb, feat_ir, tracking_feature_payload


def competition_infer(model, infer_cfg, class_names, t_rgb, t_ir, t_prev_rgb, t_prev_ir, tracking_cfg=None):
    base_size = t_rgb.shape[-1]
    transforms = build_tta_transforms(infer_cfg, base_size)
    prediction_sets = []
    heatmap_feats = None
    tracking_source = None

    for transform_cfg in transforms:
        aug_rgb, aug_ir, aug_prev_rgb, aug_prev_ir = apply_tta_transforms(
            t_rgb,
            t_ir,
            t_prev_rgb,
            t_prev_ir,
            transform_cfg,
        )
        preds, feat_rgb, feat_ir, tracking_feature_payload = run_single_prediction(
            model,
            infer_cfg,
            class_names,
            aug_rgb,
            aug_ir,
            aug_prev_rgb,
            aug_prev_ir,
            tracking_cfg=tracking_cfg,
        )
        prediction_sets.append(invert_tta_predictions(preds, transform_cfg, base_size))
        if heatmap_feats is None:
            heatmap_feats = (feat_rgb, feat_ir)
        if tracking_feature_payload is not None:
            if tracking_source is None:
                tracking_source = {'payload': tracking_feature_payload, 'transform_cfg': transform_cfg}
            if int(transform_cfg.get('size', base_size)) == int(base_size) and not transform_cfg.get('horizontal_flip', False):
                tracking_source = {'payload': tracking_feature_payload, 'transform_cfg': transform_cfg}

    merged = merge_obb_predictions(
        prediction_sets,
        method=infer_cfg['merge']['method'],
        iou_threshold=infer_cfg['merge']['iou_threshold'],
        max_det=infer_cfg['merge']['max_det'],
    )
    appearance_payload = maybe_extract_detection_appearance_features(
        tracking_source['payload'] if tracking_source is not None else None,
        merged,
        tracking_cfg,
        transform_cfg=tracking_source['transform_cfg'] if tracking_source is not None else None,
        base_size=base_size,
    )
    reliability_payload = maybe_extract_detection_reliability_features(
        tracking_source['payload'] if tracking_source is not None else None,
        merged,
        tracking_cfg,
        transform_cfg=tracking_source['transform_cfg'] if tracking_source is not None else None,
        base_size=base_size,
    )
    feature_assist_payload = maybe_extract_detection_feature_assist_features(
        tracking_source['payload'] if tracking_source is not None else None,
        merged,
        tracking_cfg,
        transform_cfg=tracking_source['transform_cfg'] if tracking_source is not None else None,
        base_size=base_size,
    )
    return merged, heatmap_feats, appearance_payload, reliability_payload, feature_assist_payload


def collect_frame_pairs(source_rgb, source_ir):
    rgb_path = Path(source_rgb)
    ir_path = Path(source_ir)

    if not rgb_path.exists():
        raise FileNotFoundError(f'RGB source does not exist: {rgb_path}')
    if not ir_path.exists():
        raise FileNotFoundError(f'IR source does not exist: {ir_path}')

    if rgb_path.is_dir() and ir_path.is_dir():
        frame_pairs = []
        rgb_files = sorted([path for path in rgb_path.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES])
        for rgb_file in rgb_files:
            paired_ir = ir_path / rgb_file.name
            if paired_ir.exists() and paired_ir.suffix.lower() in IMAGE_SUFFIXES:
                frame_pairs.append((rgb_file, paired_ir))
        if not frame_pairs:
            raise FileNotFoundError(
                f'No aligned RGB/IR image pairs found under directories: {rgb_path} | {ir_path}'
            )
        return frame_pairs

    if rgb_path.is_file() and ir_path.is_file():
        return [(rgb_path, ir_path)]

    raise FileNotFoundError('source_rgb/source_ir must both be files or both be directories with aligned filenames.')


def prepare_inputs(cfg, device, img_rgb, img_ir, prev_rgb_img, prev_ir_img):
    img_rgb_pad, r, (dw, dh) = letterbox(img_rgb, new_shape=(cfg.dataset.imgsz, cfg.dataset.imgsz))
    img_ir_pad, _, _ = letterbox(img_ir, new_shape=(cfg.dataset.imgsz, cfg.dataset.imgsz))
    prev_rgb_pad, _, _ = letterbox(prev_rgb_img, new_shape=(cfg.dataset.imgsz, cfg.dataset.imgsz))
    prev_ir_pad, _, _ = letterbox(prev_ir_img, new_shape=(cfg.dataset.imgsz, cfg.dataset.imgsz))

    t_rgb = to_tensor(img_rgb_pad, swap_rb=True).to(device)
    t_ir = to_tensor(img_ir_pad, swap_rb=False).to(device)
    t_prev_rgb = to_tensor(prev_rgb_pad, swap_rb=True).to(device)
    t_prev_ir = to_tensor(prev_ir_pad, swap_rb=False).to(device)
    return t_rgb, t_ir, t_prev_rgb, t_prev_ir, r, dw, dh


def project_predictions_to_original(preds, scale_ratio, dw, dh):
    if len(preds) == 0:
        return preds.detach().clone().cpu()
    preds = preds.detach().clone().cpu()
    preds[:, 0] -= dw
    preds[:, 1] -= dh
    preds[:, :4] /= scale_ratio
    return preds


def draw_result_records(image, results, class_names):
    if not results:
        return image
    obb_boxes = torch.tensor([result['obb'] for result in results], dtype=torch.float32)
    scores = torch.tensor([result['score'] for result in results], dtype=torch.float32)
    classes = torch.tensor([result['class_id'] for result in results], dtype=torch.float32)
    track_ids = [result.get('track_id') for result in results]
    states = [result.get('state') for result in results]
    return draw_obb(image, obb_boxes, scores, classes, class_names, track_ids=track_ids, states=states)


def resolve_tracking_scene_context(tracking_cfg):
    modality_cfg = tracking_cfg.get('modality', {}) if isinstance(tracking_cfg, dict) else {}
    raw_scene_context = modality_cfg.get('scene_context', {}) if isinstance(modality_cfg.get('scene_context', {}), dict) else {}
    scene_context = {}
    for key in SCENE_CONTEXT_KEYS:
        value = raw_scene_context.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            scene_context[key] = text
    return scene_context


def build_tracking_frame_meta(frame_index, rgb_path, sequence_mode=False, scene_context=None):
    path = Path(rgb_path)
    frame_meta = {
        'frame_index': frame_index,
        'image_id': path.name,
        'sequence_id': path.parent.name if sequence_mode else 'single_sequence',
    }
    if scene_context:
        frame_meta.update(dict(scene_context))
    return frame_meta


def main():
    parser = argparse.ArgumentParser(description='Run RGB/IR OBB inference with optional tracking refinement.')
    parser.add_argument('--config', type=str, default='configs/default.yaml', help='Path to the config file.')
    parser.add_argument('--weights', type=str, required=True, help='Checkpoint to load for inference.')
    parser.add_argument('--source_rgb', type=str, required=True, help='RGB image file or RGB frame directory.')
    parser.add_argument('--source_ir', type=str, required=True, help='IR image file or IR frame directory.')
    parser.add_argument('--prev_rgb', type=str, default='', help='Optional previous RGB frame used to bootstrap temporal inference.')
    parser.add_argument('--prev_ir', type=str, default='', help='Optional previous IR frame used to bootstrap temporal inference.')
    parser.add_argument('--save_dir', type=str, default='outputs/detect', help='Directory to write visualizations and optional tracking JSON.')
    parser.add_argument('--heatmap', action='store_true', help='Save the first available attention heatmap overlay.')
    parser.add_argument('--device', type=int, default=0, help='CUDA device id for detection inference.')
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg, run_name = apply_experiment_runtime_overrides(cfg, config_path=args.config)
    device = resolve_detection_device(args.device)
    infer_cfg = normalize_infer_cfg(cfg.get('infer', {}), default_imgsz=cfg.dataset.imgsz, nms_cfg=cfg.val.nms)
    tracking_cfg = normalize_tracking_cfg(cfg.get('tracking', {}))

    print(f'Experiment name: {run_name}')
    print(f'Infer mode: {infer_cfg["mode"]}')
    print(f'TTA: {describe_tta_settings(infer_cfg)}')
    print(f'Classwise thresholds: {describe_classwise_thresholds(infer_cfg.get("classwise_conf_thresholds", {}))}')
    print(f'Tracking enabled: {tracking_cfg["enabled"]}')
    if tracking_cfg['enabled']:
        print(f"Appearance association enabled: {tracking_cfg['association']['use_appearance'] and tracking_cfg['appearance']['enabled']}")
        scene_context = resolve_tracking_scene_context(tracking_cfg)
        scene_adaptive_requested = bool(
            tracking_cfg.get('modality', {}).get('use_scene_adaptation', False)
            and tracking_cfg.get('association', {}).get('use_modality_awareness', False)
            and tracking_cfg.get('association', {}).get('dynamic_weighting', False)
        )
        if scene_adaptive_requested and scene_context:
            scene_desc = ', '.join(f'{key}={scene_context[key]}' for key in SCENE_CONTEXT_KEYS if key in scene_context)
            print(f'Scene-adaptive weighting: enabled ({scene_desc})')
        elif scene_adaptive_requested:
            print('Scene-adaptive weighting: enabled but no scene context provided; falling back to reliability-only dynamic weighting.')
        else:
            print('Scene-adaptive weighting: off')
    else:
        scene_context = {}

    model = build_model(cfg.model).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    frame_pairs = collect_frame_pairs(args.source_rgb, args.source_ir)
    sequence_mode = len(frame_pairs) > 1

    tracker = build_tracker_from_cfg(tracking_cfg, class_names=cfg.dataset.class_names)
    refiner = TrackAwareRefiner(tracking_cfg) if tracking_cfg.get('enabled', False) and tracking_cfg.get('refinement', {}).get('enabled', False) else None
    if tracking_cfg['enabled'] and not sequence_mode:
        print('Tracking is enabled on a single frame. Stage-2 appearance cues will still be extracted, but temporal recovery mainly helps on multi-frame inputs.')

    frame_save_dir = save_dir / 'frames' if sequence_mode else save_dir
    frame_save_dir.mkdir(parents=True, exist_ok=True)

    prev_rgb_bootstrap = cv2.imread(args.prev_rgb) if args.prev_rgb and Path(args.prev_rgb).is_file() else None
    prev_ir_bootstrap = cv2.imread(args.prev_ir) if args.prev_ir and Path(args.prev_ir).is_file() else None
    tracking_records = []

    prev_rgb_image = None
    prev_ir_image = None
    for frame_index, (rgb_path, ir_path) in enumerate(frame_pairs):
        img_rgb = cv2.imread(str(rgb_path))
        img_ir = cv2.imread(str(ir_path))
        if img_rgb is None or img_ir is None:
            continue

        if prev_rgb_image is None:
            prev_rgb_image = prev_rgb_bootstrap if prev_rgb_bootstrap is not None else img_rgb
            prev_ir_image = prev_ir_bootstrap if prev_ir_bootstrap is not None else img_ir

        t_rgb, t_ir, t_prev_rgb, t_prev_ir, scale_ratio, dw, dh = prepare_inputs(
            cfg,
            device,
            img_rgb,
            img_ir,
            prev_rgb_image,
            prev_ir_image,
        )
        preds, heatmap_feats, appearance_payload, reliability_payload, feature_assist_payload = competition_infer(
            model,
            infer_cfg,
            cfg.dataset.class_names,
            t_rgb,
            t_ir,
            t_prev_rgb,
            t_prev_ir,
            tracking_cfg=tracking_cfg,
        )
        preds = project_predictions_to_original(preds, scale_ratio, dw, dh)
        frame_meta = build_tracking_frame_meta(
            frame_index,
            rgb_path,
            sequence_mode=sequence_mode,
            scene_context=scene_context,
        )
        refinement_payload = None
        refinement_summary = None
        if refiner is not None:
            preds, appearance_payload, reliability_payload, refinement_payload, refinement_summary = refiner.refine(
                preds,
                tracker.tracks if tracker is not None else [],
                base_score_threshold=infer_cfg['conf_threshold'],
                appearance_features=appearance_payload,
                reliability_features=reliability_payload,
                feature_assist_features=feature_assist_payload,
                frame_meta=frame_meta,
            )
            feature_assist_payload = refiner.last_feature_assist_payload

        if tracker is not None:
            frame_results = tracker.update(
                preds,
                frame_meta=frame_meta,
                appearance_features=appearance_payload,
                reliability_features=reliability_payload,
                feature_assist_features=feature_assist_payload,
                refinement_payload=refinement_payload,
            )
            tracking_records.append(
                {
                    'frame_index': frame_index,
                    'image_id': rgb_path.name,
                    'results': frame_results,
                    'refinement_summary': refinement_summary or {},
                    'advanced_summary': tracker.last_frame_summary or {},
                }
            )
            canvas = draw_result_records(img_rgb.copy(), frame_results, cfg.dataset.class_names)
        else:
            frame_results = detections_to_results(preds)
            canvas = draw_result_records(img_rgb.copy(), frame_results, cfg.dataset.class_names)

        output_name = f'{frame_index:06d}.jpg' if sequence_mode else 'inference_result.jpg'
        cv2.imwrite(str(frame_save_dir / output_name), canvas)

        if args.heatmap and heatmap_feats is not None and heatmap_feats[1] is not None:
            heatmap_name = f'p2_attention_heatmap_{frame_index:06d}.jpg' if sequence_mode else 'p2_attention_heatmap.jpg'
            draw_heatmap(img_rgb, heatmap_feats[1][0], save_dir / heatmap_name)

        prev_rgb_image = img_rgb
        prev_ir_image = img_ir

    if tracker is not None:
        export_path = save_dir / 'tracking_results.json'
        with export_path.open('w', encoding='utf-8') as file:
            json.dump(tracking_records, file, ensure_ascii=False, indent=2)
        print(f'Tracking results saved to {export_path}')
    else:
        print(f'Detection results saved to {save_dir}')


if __name__ == '__main__':
    main()
