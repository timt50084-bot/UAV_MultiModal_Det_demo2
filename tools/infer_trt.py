import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

if __package__ in {None, ''}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.deployment.tensorrt_runtime import TensorRTEngineRunner, TensorRTRuntimeError  # noqa: E402
from src.model.bbox_utils import non_max_suppression_obb, xywhr2xyxyxyxy  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.config_utils import apply_experiment_runtime_overrides  # noqa: E402
from src.utils.postprocess_tuning import normalize_infer_cfg  # noqa: E402


COLORS = [(0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255), (0, 0, 255)]
IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            'Run minimal TensorRT RGB/IR OBB inference. '
            'Workflow: tools/export.py -> tools/build_trt_engine.py -> tools/infer_trt.py'
        )
    )
    parser.add_argument('--config', type=str, default='configs/default.yaml', help='Path to the config file.')
    parser.add_argument('--engine', type=str, required=True, help='TensorRT engine file generated from the ONNX export.')
    parser.add_argument('--source_rgb', type=str, required=True, help='RGB image file or aligned RGB frame directory.')
    parser.add_argument('--source_ir', type=str, required=True, help='IR image file or aligned IR frame directory.')
    parser.add_argument('--save_dir', type=str, default='outputs/detect_trt', help='Directory to save visualizations and JSON results.')
    parser.add_argument('--device', type=int, default=0, help='CUDA device id for TensorRT inference.')
    return parser


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


def to_tensor(image_bgr, swap_rb=False):
    image = image_bgr[..., ::-1] if swap_rb else image_bgr
    return torch.from_numpy(image.transpose(2, 0, 1).copy()).float().unsqueeze(0) / 255.0


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


def preprocess_pair(img_rgb, img_ir, imgsz):
    img_rgb_pad, r, (dw, dh) = letterbox(img_rgb, new_shape=(imgsz, imgsz))
    img_ir_pad, _, _ = letterbox(img_ir, new_shape=(imgsz, imgsz))
    t_rgb = to_tensor(img_rgb_pad, swap_rb=True)
    t_ir = to_tensor(img_ir_pad, swap_rb=False)
    return t_rgb, t_ir, r, dw, dh


def postprocess_flat_predictions(flat_preds, conf_thres, iou_thres, max_det):
    flat_preds = torch.nan_to_num(flat_preds.float(), nan=0.0, posinf=1e4, neginf=-1e4)
    flat_preds[..., :5] = flat_preds[..., :5].clamp(-1e4, 1e4)
    if flat_preds.shape[1] > 3000:
        _, idx = flat_preds[..., 5:].amax(dim=-1).topk(3000, dim=-1)
        flat_preds = flat_preds.gather(1, idx.unsqueeze(-1).expand(-1, -1, flat_preds.shape[-1]))
    flat_preds[..., 5:] = flat_preds[..., 5:].sigmoid()
    return non_max_suppression_obb(
        flat_preds,
        conf_thres=conf_thres,
        iou_thres=iou_thres,
        max_det=max_det,
    )[0]


def project_predictions_to_original(preds, scale_ratio, dw, dh):
    if len(preds) == 0:
        return preds.detach().clone().cpu()
    preds = preds.detach().clone().cpu()
    preds[:, 0] -= dw
    preds[:, 1] -= dh
    preds[:, :4] /= scale_ratio
    return preds


def draw_obb(image, obb_boxes, scores, classes, class_names):
    polygons = xywhr2xyxyxyxy(obb_boxes).cpu().numpy()
    for i, poly in enumerate(polygons):
        cls_id = int(classes[i].item())
        class_name = class_names[cls_id] if 0 <= cls_id < len(class_names) else str(cls_id)
        label = f"{class_name} {scores[i].item():.2f}"
        color = COLORS[cls_id % len(COLORS)]
        poly = np.int32(poly)
        cv2.polylines(image, [poly], isClosed=True, color=color, thickness=2, lineType=cv2.LINE_AA)
        pt1 = tuple(poly[0])
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
        cv2.rectangle(image, pt1, (pt1[0] + t_size[0], pt1[1] - t_size[1] - 3), color, -1, cv2.LINE_AA)
        cv2.putText(image, label, (pt1[0], pt1[1] - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return image


def detections_to_json_records(preds, class_names):
    results = []
    for det in preds.tolist():
        cx, cy, w, h, angle, score, class_id = det
        class_id = int(class_id)
        results.append({
            'obb': [cx, cy, w, h, angle],
            'score': score,
            'class_id': class_id,
            'class_name': class_names[class_id] if 0 <= class_id < len(class_names) else str(class_id),
        })
    return results


def main(argv=None):
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)
    cfg, run_name = apply_experiment_runtime_overrides(cfg, config_path=args.config)
    infer_cfg = normalize_infer_cfg(cfg.get('infer', {}), default_imgsz=cfg.dataset.imgsz, nms_cfg=cfg.val.nms)

    print(f'Experiment name: {run_name}')
    print('TensorRT runtime mode: raw engine forward + Python-side OBB postprocess')
    print('Unsupported in this minimal TensorRT path: tracking, TTA, classwise thresholds, heatmaps')

    runner = TensorRTEngineRunner(args.engine, device_id=args.device)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    frame_pairs = collect_frame_pairs(args.source_rgb, args.source_ir)
    sequence_mode = len(frame_pairs) > 1
    frame_save_dir = save_dir / 'frames' if sequence_mode else save_dir
    frame_save_dir.mkdir(parents=True, exist_ok=True)

    json_records = []
    for frame_index, (rgb_path, ir_path) in enumerate(frame_pairs):
        img_rgb = cv2.imread(str(rgb_path))
        img_ir = cv2.imread(str(ir_path))
        if img_rgb is None or img_ir is None:
            continue

        t_rgb, t_ir, scale_ratio, dw, dh = preprocess_pair(img_rgb, img_ir, cfg.dataset.imgsz)
        outputs = runner.infer(t_rgb, t_ir)
        flat_preds = outputs[runner.primary_output_name].float().cpu()
        preds = postprocess_flat_predictions(
            flat_preds,
            conf_thres=float(infer_cfg['conf_threshold']),
            iou_thres=float(infer_cfg['iou_threshold']),
            max_det=int(infer_cfg['merge']['max_det']),
        )
        preds = project_predictions_to_original(preds, scale_ratio, dw, dh)

        if len(preds):
            canvas = draw_obb(
                img_rgb.copy(),
                preds[:, :5],
                preds[:, 5],
                preds[:, 6],
                cfg.dataset.class_names,
            )
        else:
            canvas = img_rgb.copy()

        output_name = f'{frame_index:06d}.jpg' if sequence_mode else 'trt_inference_result.jpg'
        cv2.imwrite(str(frame_save_dir / output_name), canvas)
        json_records.append({
            'frame_index': frame_index,
            'image_id': Path(rgb_path).name,
            'results': detections_to_json_records(preds, cfg.dataset.class_names),
        })

    json_path = save_dir / 'trt_results.json'
    with json_path.open('w', encoding='utf-8') as file:
        json.dump(json_records, file, ensure_ascii=False, indent=2)
    print(f'TensorRT inference results saved to {save_dir}')


if __name__ == '__main__':
    try:
        main()
    except (TensorRTRuntimeError, FileNotFoundError, ValueError) as exc:
        print(f'[TensorRT Infer Error] {exc}', file=sys.stderr)
        raise SystemExit(1)
