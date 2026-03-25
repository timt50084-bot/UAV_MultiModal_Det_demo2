import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.amp import autocast

from src.model.bbox_utils import non_max_suppression_obb, xywhr2xyxyxyxy
from src.model.builder import build_model
from src.model.output_adapter import flatten_predictions
from src.tracking.obb_tracker import OBBTrackletManager
from src.utils.config import load_config

COLORS = [(0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255), (0, 0, 255)]


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


def draw_tracks(image, tracks, class_names):
    if not tracks:
        return image

    boxes = torch.stack([track.box for track in tracks], dim=0)
    polygons = xywhr2xyxyxyxy(boxes).cpu().numpy()

    for track, poly in zip(tracks, polygons):
        cls_id = int(track.cls_id)
        label = f"ID{track.track_id} {class_names[cls_id]} {track.score:.2f}"
        color = COLORS[track.track_id % len(COLORS)]
        poly = np.int32(poly)
        cv2.polylines(image, [poly], isClosed=True, color=color, thickness=2, lineType=cv2.LINE_AA)
        pt1 = tuple(poly[0])
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0]
        cv2.rectangle(image, pt1, (pt1[0] + t_size[0], pt1[1] - t_size[1] - 3), color, -1, cv2.LINE_AA)
        cv2.putText(image, label, (pt1[0], pt1[1] - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return image


def run_model(model, cfg, device, rgb_image, ir_image, prev_rgb_image, prev_ir_image):
    rgb_pad, r, (dw, dh) = letterbox(rgb_image, new_shape=(cfg.dataset.imgsz, cfg.dataset.imgsz))
    ir_pad, _, _ = letterbox(ir_image, new_shape=(cfg.dataset.imgsz, cfg.dataset.imgsz))
    prev_rgb_pad, _, _ = letterbox(prev_rgb_image, new_shape=(cfg.dataset.imgsz, cfg.dataset.imgsz))
    prev_ir_pad, _, _ = letterbox(prev_ir_image, new_shape=(cfg.dataset.imgsz, cfg.dataset.imgsz))

    tensor_rgb = torch.from_numpy(rgb_pad[..., ::-1].transpose(2, 0, 1).copy()).float().unsqueeze(0).to(device) / 255.0
    tensor_ir = torch.from_numpy(ir_pad.transpose(2, 0, 1).copy()).float().unsqueeze(0).to(device) / 255.0
    tensor_prev_rgb = torch.from_numpy(prev_rgb_pad[..., ::-1].transpose(2, 0, 1).copy()).float().unsqueeze(0).to(device) / 255.0
    tensor_prev_ir = torch.from_numpy(prev_ir_pad.transpose(2, 0, 1).copy()).float().unsqueeze(0).to(device) / 255.0

    amp_enabled = device.type == 'cuda'
    with torch.no_grad(), autocast(device_type=device.type, enabled=amp_enabled):
        outputs, _, _ = model(tensor_rgb, tensor_ir, prev_rgb=tensor_prev_rgb, prev_ir=tensor_prev_ir)
        flat_preds, _ = flatten_predictions(outputs)
        flat_preds = torch.nan_to_num(flat_preds, nan=0.0, posinf=1e4, neginf=-1e4)
        flat_preds[..., :5] = flat_preds[..., :5].clamp(-1e4, 1e4)
        flat_preds[..., 5:] = flat_preds[..., 5:].sigmoid()

    preds = non_max_suppression_obb(flat_preds.float(), **cfg.val.nms)[0]
    if len(preds) == 0:
        return preds, r, dw, dh

    obb_boxes = preds[:, :5].clone()
    obb_boxes[:, 0] -= dw
    obb_boxes[:, 1] -= dh
    obb_boxes[:, :4] /= r

    preds = preds.clone()
    preds[:, :5] = obb_boxes
    return preds.cpu(), r, dw, dh


def main():
    parser = argparse.ArgumentParser(description="Track RGB-IR frame sequences with OBB detections.")
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    parser.add_argument('--weights', type=str, required=True)
    parser.add_argument('--source_rgb_dir', type=str, required=True)
    parser.add_argument('--source_ir_dir', type=str, required=True)
    parser.add_argument('--save_dir', type=str, default='outputs/track')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--track_iou', type=float, default=0.2)
    parser.add_argument('--max_age', type=int, default=15)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device('cpu' if args.device < 0 or not torch.cuda.is_available() else f'cuda:{args.device}')

    model = build_model(cfg.model).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()

    tracker = OBBTrackletManager(iou_threshold=args.track_iou, max_age=args.max_age)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    rgb_files = sorted([p for p in Path(args.source_rgb_dir).iterdir() if p.suffix.lower() in {'.jpg', '.png', '.jpeg', '.bmp'}])
    prev_rgb_image, prev_ir_image = None, None

    for frame_idx, rgb_path in enumerate(rgb_files):
        ir_path = Path(args.source_ir_dir) / rgb_path.name
        if not ir_path.exists():
            continue

        rgb_image = cv2.imread(str(rgb_path))
        ir_image = cv2.imread(str(ir_path))
        if rgb_image is None or ir_image is None:
            continue

        if prev_rgb_image is None or prev_ir_image is None:
            prev_rgb_image, prev_ir_image = rgb_image, ir_image

        detections, _, _, _ = run_model(model, cfg, device, rgb_image, ir_image, prev_rgb_image, prev_ir_image)
        tracks = tracker.update(detections if len(detections) > 0 else torch.zeros((0, 7), dtype=torch.float32))
        prev_rgb_image, prev_ir_image = rgb_image, ir_image

        canvas = draw_tracks(rgb_image.copy(), tracks, cfg.dataset.class_names)
        cv2.imwrite(str(save_dir / f"{frame_idx:06d}.jpg"), canvas)

        txt_path = save_dir / f"{frame_idx:06d}.txt"
        with open(txt_path, 'w') as f:
            for track in tracks:
                box = track.box.tolist()
                f.write(
                    f"{track.track_id} {track.cls_id} {track.score:.6f} "
                    f"{box[0]:.6f} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f} {box[4]:.6f}\n"
                )

    print(f"Tracking results saved to {os.path.abspath(args.save_dir)}")


if __name__ == '__main__':
    main()
