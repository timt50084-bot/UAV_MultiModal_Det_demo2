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
VIDEO_SUFFIXES = {'.mp4', '.avi', '.mov', '.mkv', '.mpeg', '.mpg', '.m4v', '.wmv'}
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


def ensure_color_image(image):
    if image is None:
        return None
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim == 3 and image.shape[2] == 1:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image


def build_pseudo_ir_image(image_bgr):
    image_bgr = ensure_color_image(image_bgr)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def build_missing_modality(image_bgr, fallback, missing_modality):
    image_bgr = ensure_color_image(image_bgr)
    fallback = str(fallback).lower()
    if fallback == 'replicate':
        return image_bgr.copy()
    if fallback == 'zero':
        return np.zeros_like(image_bgr)
    if fallback == 'grayscale_to_ir':
        if missing_modality == 'ir':
            return build_pseudo_ir_image(image_bgr)
        return image_bgr.copy()
    raise ValueError(f'Unsupported single_input_fallback: {fallback}')


def build_single_input_modalities(image_bgr, input_mode, fallback):
    image_bgr = ensure_color_image(image_bgr)
    input_mode = str(input_mode).lower()
    if input_mode == 'single_rgb':
        return image_bgr.copy(), build_missing_modality(image_bgr, fallback, missing_modality='ir')
    if input_mode == 'single_ir':
        return build_missing_modality(image_bgr, fallback, missing_modality='rgb'), image_bgr.copy()
    raise ValueError(f'Unsupported single input mode: {input_mode}')


def classify_single_source(source):
    source_path = Path(source)
    if not source_path.exists():
        raise FileNotFoundError(f'Source does not exist: {source_path}')
    if not source_path.is_file():
        raise ValueError(f'Single-input source must be an image or video file, but got: {source_path}')
    suffix = source_path.suffix.lower()
    if suffix in IMAGE_SUFFIXES:
        return 'image'
    if suffix in VIDEO_SUFFIXES:
        return 'video'
    raise ValueError(
        f'Unsupported --source file type: {source_path}. '
        f'Supported images: {sorted(IMAGE_SUFFIXES)}; videos: {sorted(VIDEO_SUFFIXES)}.'
    )


def resolve_input_spec(args):
    source = str(args.source or '').strip()
    source_rgb = str(args.source_rgb or '').strip()
    source_ir = str(args.source_ir or '').strip()
    input_mode = str(args.input_mode).lower()

    if input_mode not in {'auto', 'paired', 'single_rgb', 'single_ir'}:
        raise ValueError(f'Unsupported input_mode: {input_mode}')

    if input_mode == 'paired':
        if source:
            raise ValueError('--input_mode paired does not accept --source. Use --source_rgb and --source_ir instead.')
        if not (source_rgb and source_ir):
            raise ValueError('Paired inference requires both --source_rgb and --source_ir.')
        return {
            'mode': 'paired',
            'source_rgb': source_rgb,
            'source_ir': source_ir,
            'single_source': '',
            'single_mode': '',
            'assumed_single_mode': False,
        }

    if input_mode in {'single_rgb', 'single_ir'}:
        if source and (source_rgb or source_ir):
            raise ValueError('Single-input inference accepts either --source or one legacy modality flag, not both.')
        preferred_source = source_rgb if input_mode == 'single_rgb' else source_ir
        other_source = source_ir if input_mode == 'single_rgb' else source_rgb
        single_source = source or preferred_source
        if not single_source:
            expected_flag = '--source_rgb' if input_mode == 'single_rgb' else '--source_ir'
            raise ValueError(f'{input_mode} inference requires --source or {expected_flag}.')
        if other_source:
            raise ValueError(f'{input_mode} inference only accepts one real modality source; remove the other paired flag.')
        return {
            'mode': 'single',
            'source_rgb': '',
            'source_ir': '',
            'single_source': single_source,
            'single_mode': input_mode,
            'assumed_single_mode': False,
        }

    if source:
        if source_rgb or source_ir:
            raise ValueError('Use either --source or --source_rgb/--source_ir, not both.')
        return {
            'mode': 'single',
            'source_rgb': '',
            'source_ir': '',
            'single_source': source,
            'single_mode': 'single_rgb',
            'assumed_single_mode': True,
        }
    if source_rgb and source_ir:
        return {
            'mode': 'paired',
            'source_rgb': source_rgb,
            'source_ir': source_ir,
            'single_source': '',
            'single_mode': '',
            'assumed_single_mode': False,
        }
    if source_rgb:
        return {
            'mode': 'single',
            'source_rgb': '',
            'source_ir': '',
            'single_source': source_rgb,
            'single_mode': 'single_rgb',
            'assumed_single_mode': False,
        }
    if source_ir:
        return {
            'mode': 'single',
            'source_rgb': '',
            'source_ir': '',
            'single_source': source_ir,
            'single_mode': 'single_ir',
            'assumed_single_mode': False,
        }
    raise ValueError('Provide --source for single-input inference, or provide both --source_rgb and --source_ir for paired inference.')


def resolve_single_prev_source_path(input_mode, prev_rgb, prev_ir):
    if str(input_mode).lower() == 'single_ir':
        candidates = [prev_ir, prev_rgb]
    else:
        candidates = [prev_rgb, prev_ir]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return candidate
    return ''


def open_video_capture(video_path):
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f'Failed to open video source: {video_path}')
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 0:
        fps = 25.0
    return capture, fps


def iter_video_capture_frames(capture):
    frame_index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        yield frame_index, ensure_color_image(frame)
        frame_index += 1


def iter_video_frames(video_path):
    capture, _ = open_video_capture(video_path)
    try:
        for item in iter_video_capture_frames(capture):
            yield item
    finally:
        capture.release()


def create_video_writer(save_dir, source_path, fps, frame_size):
    source_path = Path(source_path)
    width, height = int(frame_size[0]), int(frame_size[1])
    candidates = [
        ('mp4v', save_dir / f'{source_path.stem}_result.mp4'),
        ('avc1', save_dir / f'{source_path.stem}_result.mp4'),
        ('XVID', save_dir / f'{source_path.stem}_result.avi'),
        ('MJPG', save_dir / f'{source_path.stem}_result.avi'),
    ]
    for fourcc_name, output_path in candidates:
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*fourcc_name),
            float(fps),
            (width, height),
        )
        if writer.isOpened():
            return writer, output_path
        writer.release()
    raise RuntimeError(f'Failed to create an output video writer under: {save_dir}')


def load_model_weights(model, weights_path, device):
    checkpoint = torch.load(weights_path, map_location=device)
    if isinstance(checkpoint, dict):
        if 'state_dict' in checkpoint and isinstance(checkpoint['state_dict'], dict):
            checkpoint = checkpoint['state_dict']
        elif 'model' in checkpoint and isinstance(checkpoint['model'], dict):
            checkpoint = checkpoint['model']
    model.load_state_dict(checkpoint)


def run_inference_frame(model, cfg, infer_cfg, tracking_cfg, device, class_names,
                        img_rgb, img_ir, prev_rgb_image, prev_ir_image):
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
        class_names,
        t_rgb,
        t_ir,
        t_prev_rgb,
        t_prev_ir,
        tracking_cfg=tracking_cfg,
    )
    preds = project_predictions_to_original(preds, scale_ratio, dw, dh)
    return preds, heatmap_feats, appearance_payload, reliability_payload, feature_assist_payload


def process_inference_frame(model, cfg, infer_cfg, tracking_cfg, device, tracker, refiner, class_names,
                            scene_context, frame_index, frame_identifier, sequence_mode, canvas_image,
                            img_rgb, img_ir, prev_rgb_image, prev_ir_image):
    preds, heatmap_feats, appearance_payload, reliability_payload, feature_assist_payload = run_inference_frame(
        model,
        cfg,
        infer_cfg,
        tracking_cfg,
        device,
        class_names,
        img_rgb,
        img_ir,
        prev_rgb_image,
        prev_ir_image,
    )
    frame_meta = build_tracking_frame_meta(
        frame_index,
        frame_identifier,
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

    tracking_record = None
    if tracker is not None:
        frame_results = tracker.update(
            preds,
            frame_meta=frame_meta,
            appearance_features=appearance_payload,
            reliability_features=reliability_payload,
            feature_assist_features=feature_assist_payload,
            refinement_payload=refinement_payload,
        )
        tracking_record = {
            'frame_index': frame_index,
            'image_id': Path(frame_identifier).name,
            'results': frame_results,
            'refinement_summary': refinement_summary or {},
            'advanced_summary': tracker.last_frame_summary or {},
        }
    else:
        frame_results = detections_to_results(preds)

    canvas = draw_result_records(canvas_image.copy(), frame_results, class_names)
    return canvas, heatmap_feats, tracking_record


def ensure_paired_sources_supported(source_rgb, source_ir):
    rgb_path = Path(source_rgb)
    ir_path = Path(source_ir)
    if rgb_path.is_file() and ir_path.is_file():
        rgb_video = rgb_path.suffix.lower() in VIDEO_SUFFIXES
        ir_video = ir_path.suffix.lower() in VIDEO_SUFFIXES
        if rgb_video or ir_video:
            raise ValueError(
                'Paired video files are not supported in tools/infer.py. '
                'Use extracted frame directories for paired RGB/IR inference, or use --source for single-input video fallback.'
            )


def run_paired_inference(args, model, cfg, infer_cfg, tracking_cfg, device, tracker, refiner, scene_context, save_dir):
    ensure_paired_sources_supported(args.source_rgb, args.source_ir)
    frame_pairs = collect_frame_pairs(args.source_rgb, args.source_ir)
    sequence_mode = len(frame_pairs) > 1
    if tracking_cfg['enabled'] and not sequence_mode:
        print('Tracking is enabled on a single frame. Stage-2 appearance cues will still be extracted, but temporal recovery mainly helps on multi-frame inputs.')

    frame_save_dir = save_dir / 'frames' if sequence_mode else save_dir
    frame_save_dir.mkdir(parents=True, exist_ok=True)

    prev_rgb_bootstrap = ensure_color_image(cv2.imread(args.prev_rgb)) if args.prev_rgb and Path(args.prev_rgb).is_file() else None
    prev_ir_bootstrap = ensure_color_image(cv2.imread(args.prev_ir)) if args.prev_ir and Path(args.prev_ir).is_file() else None

    tracking_records = []
    prev_rgb_image = None
    prev_ir_image = None
    for frame_index, (rgb_path, ir_path) in enumerate(frame_pairs):
        img_rgb = ensure_color_image(cv2.imread(str(rgb_path)))
        img_ir = ensure_color_image(cv2.imread(str(ir_path)))
        if img_rgb is None or img_ir is None:
            continue

        if prev_rgb_image is None:
            prev_rgb_image = prev_rgb_bootstrap if prev_rgb_bootstrap is not None else img_rgb
            prev_ir_image = prev_ir_bootstrap if prev_ir_bootstrap is not None else img_ir

        canvas, heatmap_feats, tracking_record = process_inference_frame(
            model=model,
            cfg=cfg,
            infer_cfg=infer_cfg,
            tracking_cfg=tracking_cfg,
            device=device,
            tracker=tracker,
            refiner=refiner,
            class_names=cfg.dataset.class_names,
            scene_context=scene_context,
            frame_index=frame_index,
            frame_identifier=rgb_path,
            sequence_mode=sequence_mode,
            canvas_image=img_rgb,
            img_rgb=img_rgb,
            img_ir=img_ir,
            prev_rgb_image=prev_rgb_image,
            prev_ir_image=prev_ir_image,
        )

        output_name = f'{frame_index:06d}.jpg' if sequence_mode else 'inference_result.jpg'
        output_path = frame_save_dir / output_name
        cv2.imwrite(str(output_path), canvas)

        if args.heatmap and heatmap_feats is not None and heatmap_feats[1] is not None:
            heatmap_name = f'p2_attention_heatmap_{frame_index:06d}.jpg' if sequence_mode else 'p2_attention_heatmap.jpg'
            draw_heatmap(img_rgb, heatmap_feats[1][0], save_dir / heatmap_name)

        if tracking_record is not None:
            tracking_records.append(tracking_record)
        prev_rgb_image = img_rgb
        prev_ir_image = img_ir

    result_path = frame_save_dir if sequence_mode else (frame_save_dir / 'inference_result.jpg')
    return tracking_records, {'kind': 'frames' if sequence_mode else 'image', 'path': result_path}


def run_single_source_inference(args, input_spec, model, cfg, infer_cfg, tracking_cfg, device, tracker, refiner, scene_context, save_dir):
    input_mode = input_spec['single_mode']
    source_path = Path(input_spec['single_source'])
    source_kind = classify_single_source(source_path)

    if input_spec.get('assumed_single_mode', False):
        print('Warning: --source with input_mode=auto defaults to single_rgb. Use --input_mode single_ir to treat the source as infrared.')
    print(
        'Warning: single-input fallback is active '
        f'(input_mode={input_mode}, fallback={args.single_input_fallback}). '
        'This compatibility path is for inference only and does not equal true paired RGB/IR accuracy.'
    )
    if tracking_cfg['enabled'] and source_kind == 'image':
        print('Tracking is enabled on a single frame. Stage-2 appearance cues will still be extracted, but temporal recovery mainly helps on multi-frame inputs.')

    bootstrap_path = resolve_single_prev_source_path(input_mode, args.prev_rgb, args.prev_ir)
    bootstrap_image = ensure_color_image(cv2.imread(bootstrap_path)) if bootstrap_path else None
    tracking_records = []

    if source_kind == 'image':
        source_image = ensure_color_image(cv2.imread(str(source_path)))
        if source_image is None:
            raise RuntimeError(f'Failed to read image source: {source_path}')
        prev_source_image = bootstrap_image if bootstrap_image is not None else source_image
        img_rgb, img_ir = build_single_input_modalities(source_image, input_mode, args.single_input_fallback)
        prev_rgb_image, prev_ir_image = build_single_input_modalities(prev_source_image, input_mode, args.single_input_fallback)
        canvas, heatmap_feats, tracking_record = process_inference_frame(
            model=model,
            cfg=cfg,
            infer_cfg=infer_cfg,
            tracking_cfg=tracking_cfg,
            device=device,
            tracker=tracker,
            refiner=refiner,
            class_names=cfg.dataset.class_names,
            scene_context=scene_context,
            frame_index=0,
            frame_identifier=source_path,
            sequence_mode=False,
            canvas_image=source_image,
            img_rgb=img_rgb,
            img_ir=img_ir,
            prev_rgb_image=prev_rgb_image,
            prev_ir_image=prev_ir_image,
        )
        output_path = save_dir / f'{source_path.stem}_result{source_path.suffix}'
        cv2.imwrite(str(output_path), canvas)
        if args.heatmap and heatmap_feats is not None and heatmap_feats[1] is not None:
            draw_heatmap(source_image, heatmap_feats[1][0], save_dir / 'p2_attention_heatmap.jpg')
        if tracking_record is not None:
            tracking_records.append(tracking_record)
        return tracking_records, {'kind': 'image', 'path': output_path}

    capture, fps = open_video_capture(source_path)
    video_writer = None
    output_video_path = None
    frames_seen = 0
    prev_source_image = None
    try:
        for frame_index, source_frame in iter_video_capture_frames(capture):
            frames_seen += 1
            if prev_source_image is None:
                prev_source_image = bootstrap_image if bootstrap_image is not None else source_frame

            img_rgb, img_ir = build_single_input_modalities(source_frame, input_mode, args.single_input_fallback)
            prev_rgb_image, prev_ir_image = build_single_input_modalities(prev_source_image, input_mode, args.single_input_fallback)
            frame_identifier = Path(source_path.stem) / f'frame_{frame_index:06d}.jpg'
            canvas, heatmap_feats, tracking_record = process_inference_frame(
                model=model,
                cfg=cfg,
                infer_cfg=infer_cfg,
                tracking_cfg=tracking_cfg,
                device=device,
                tracker=tracker,
                refiner=refiner,
                class_names=cfg.dataset.class_names,
                scene_context=scene_context,
                frame_index=frame_index,
                frame_identifier=frame_identifier,
                sequence_mode=True,
                canvas_image=source_frame,
                img_rgb=img_rgb,
                img_ir=img_ir,
                prev_rgb_image=prev_rgb_image,
                prev_ir_image=prev_ir_image,
            )
            if video_writer is None:
                video_writer, output_video_path = create_video_writer(
                    save_dir,
                    source_path,
                    fps,
                    (canvas.shape[1], canvas.shape[0]),
                )
            video_writer.write(canvas)

            if args.heatmap and heatmap_feats is not None and heatmap_feats[1] is not None:
                draw_heatmap(source_frame, heatmap_feats[1][0], save_dir / f'p2_attention_heatmap_{frame_index:06d}.jpg')
            if tracking_record is not None:
                tracking_records.append(tracking_record)
            prev_source_image = source_frame
    finally:
        capture.release()
        if video_writer is not None:
            video_writer.release()

    if frames_seen == 0 or output_video_path is None:
        raise RuntimeError(f'No frames were decoded from video source: {source_path}')
    return tracking_records, {'kind': 'video', 'path': output_video_path}


def main():
    parser = argparse.ArgumentParser(description='Run RGB/IR OBB inference with paired or single-input fallback modes.')
    parser.add_argument('--config', type=str, default='configs/default.yaml', help='Path to the config file.')
    parser.add_argument('--weights', type=str, required=True, help='Checkpoint to load for inference.')
    parser.add_argument('--source', type=str, default='', help='Single input image or video used in fallback inference mode.')
    parser.add_argument('--source_rgb', type=str, default='', help='RGB image file or RGB frame directory.')
    parser.add_argument('--source_ir', type=str, default='', help='IR image file or IR frame directory.')
    parser.add_argument('--input_mode', type=str, default='auto', choices=['auto', 'paired', 'single_rgb', 'single_ir'], help='Input interpretation mode. Auto keeps paired RGB/IR when both sources are provided, otherwise falls back to a single source.')
    parser.add_argument('--single_input_fallback', type=str, default='grayscale_to_ir', choices=['grayscale_to_ir', 'replicate', 'zero'], help='How to synthesize the missing modality in single-input inference.')
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
    input_spec = resolve_input_spec(args)

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
    load_model_weights(model, args.weights, device)
    model.eval()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    tracker = build_tracker_from_cfg(tracking_cfg, class_names=cfg.dataset.class_names)
    refiner = TrackAwareRefiner(tracking_cfg) if tracking_cfg.get('enabled', False) and tracking_cfg.get('refinement', {}).get('enabled', False) else None

    if input_spec['mode'] == 'paired':
        tracking_records, output_summary = run_paired_inference(
            args=args,
            model=model,
            cfg=cfg,
            infer_cfg=infer_cfg,
            tracking_cfg=tracking_cfg,
            device=device,
            tracker=tracker,
            refiner=refiner,
            scene_context=scene_context,
            save_dir=save_dir,
        )
    else:
        tracking_records, output_summary = run_single_source_inference(
            args=args,
            input_spec=input_spec,
            model=model,
            cfg=cfg,
            infer_cfg=infer_cfg,
            tracking_cfg=tracking_cfg,
            device=device,
            tracker=tracker,
            refiner=refiner,
            scene_context=scene_context,
            save_dir=save_dir,
        )

    if tracker is not None:
        export_path = save_dir / 'tracking_results.json'
        with export_path.open('w', encoding='utf-8') as file:
            json.dump(tracking_records, file, ensure_ascii=False, indent=2)
        print(f'Tracking results saved to {export_path}')

    result_path = output_summary.get('path')
    if output_summary.get('kind') == 'video':
        print(f'Detection video saved to {result_path}')
    elif output_summary.get('kind') == 'image':
        print(f'Detection image saved to {result_path}')
    else:
        print(f'Detection results saved to {result_path}')


if __name__ == '__main__':
    main()
