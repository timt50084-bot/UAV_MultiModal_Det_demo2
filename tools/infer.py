import argparse
import os

import cv2
import numpy as np
import torch
from torch.amp import autocast

from src.model.bbox_utils import non_max_suppression_obb, xywhr2xyxyxyxy
from src.model.builder import build_model
from src.model.output_adapter import flatten_predictions
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


def draw_obb(image, obb_boxes, scores, classes, class_names):
    polygons = xywhr2xyxyxyxy(obb_boxes).cpu().numpy()
    for i, poly in enumerate(polygons):
        cls_id = int(classes[i].item())
        label = f"{class_names[cls_id]} {scores[i].item():.2f}"
        color = COLORS[cls_id % len(COLORS)]
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
    cv2.imwrite(save_path, cv2.addWeighted(original_img, 0.6, heatmap_color, 0.4, 0))
    print(f"Heatmap saved to {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    parser.add_argument('--weights', type=str, required=True)
    parser.add_argument('--source_rgb', type=str, required=True)
    parser.add_argument('--source_ir', type=str, required=True)
    parser.add_argument('--prev_rgb', type=str, default='')
    parser.add_argument('--prev_ir', type=str, default='')
    parser.add_argument('--save_dir', type=str, default='outputs/detect')
    parser.add_argument('--heatmap', action='store_true')
    parser.add_argument('--device', type=int, default=0)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device('cpu' if args.device < 0 or not torch.cuda.is_available() else f'cuda:{args.device}')

    model = build_model(cfg.model).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()
    os.makedirs(args.save_dir, exist_ok=True)

    img_rgb = cv2.imread(args.source_rgb)
    img_ir = cv2.imread(args.source_ir)
    prev_rgb_img = cv2.imread(args.prev_rgb) if args.prev_rgb else img_rgb
    prev_ir_img = cv2.imread(args.prev_ir) if args.prev_ir else img_ir
    img_raw = img_rgb.copy()

    img_rgb_pad, r, (dw, dh) = letterbox(img_rgb, new_shape=(cfg.dataset.imgsz, cfg.dataset.imgsz))
    img_ir_pad, _, _ = letterbox(img_ir, new_shape=(cfg.dataset.imgsz, cfg.dataset.imgsz))
    prev_rgb_pad, _, _ = letterbox(prev_rgb_img, new_shape=(cfg.dataset.imgsz, cfg.dataset.imgsz))
    prev_ir_pad, _, _ = letterbox(prev_ir_img, new_shape=(cfg.dataset.imgsz, cfg.dataset.imgsz))

    t_rgb = torch.from_numpy(img_rgb_pad[..., ::-1].transpose(2, 0, 1).copy()).float().unsqueeze(0).to(device) / 255.0
    t_ir = torch.from_numpy(img_ir_pad.transpose(2, 0, 1).copy()).float().unsqueeze(0).to(device) / 255.0
    t_prev_rgb = torch.from_numpy(prev_rgb_pad[..., ::-1].transpose(2, 0, 1).copy()).float().unsqueeze(0).to(device) / 255.0
    t_prev_ir = torch.from_numpy(prev_ir_pad.transpose(2, 0, 1).copy()).float().unsqueeze(0).to(device) / 255.0

    amp_enabled = device.type == 'cuda'
    with torch.no_grad(), autocast(device_type=device.type, enabled=amp_enabled):
        if args.heatmap:
            outputs, att_maps, feat_rgb, feat_ir = model(
                t_rgb, t_ir, prev_rgb=t_prev_rgb, prev_ir=t_prev_ir, return_attention_map=True
            )
        else:
            outputs, feat_rgb, feat_ir = model(t_rgb, t_ir, prev_rgb=t_prev_rgb, prev_ir=t_prev_ir)
            att_maps = None

        flat_preds, _ = flatten_predictions(outputs)
        flat_preds = torch.nan_to_num(flat_preds, nan=0.0, posinf=1e4, neginf=-1e4)
        flat_preds[..., :5] = flat_preds[..., :5].clamp(-1e4, 1e4)

        if flat_preds.shape[1] > 3000:
            _, idx = flat_preds[..., 5:].amax(dim=-1).topk(3000, dim=-1)
            flat_preds = flat_preds.gather(1, idx.unsqueeze(-1).expand(-1, -1, flat_preds.shape[-1]))
        flat_preds[..., 5:] = flat_preds[..., 5:].sigmoid()

    preds = non_max_suppression_obb(flat_preds.float(), **cfg.val.nms)[0]

    save_path = os.path.join(args.save_dir, "inference_result.jpg")
    if len(preds) > 0:
        obb_boxes = preds[:, :5].clone()
        obb_boxes[:, 0] -= dw
        obb_boxes[:, 1] -= dh
        obb_boxes[:, :4] /= r
        cv2.imwrite(save_path, draw_obb(img_raw, obb_boxes, preds[:, 5], preds[:, 6], cfg.dataset.class_names))
        print(f"Detected {len(preds)} objects.")
    else:
        cv2.imwrite(save_path, img_raw)
        print("No objects detected.")

    if args.heatmap and feat_ir is not None:
        draw_heatmap(img_rgb, feat_ir[0], os.path.join(args.save_dir, "p2_attention_heatmap.jpg"))


if __name__ == "__main__":
    main()
