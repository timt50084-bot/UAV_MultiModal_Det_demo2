from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np
import torch

from src.model.bbox_utils import xywhr2xyxyxyxy


def track_color(track_id):
    track_id = int(track_id) if track_id is not None else 0
    return (
        int((37 * track_id + 53) % 255),
        int((97 * track_id + 91) % 255),
        int((17 * track_id + 193) % 255),
    )


def _resolve_label(class_id, class_names):
    class_id = int(class_id)
    if class_names and 0 <= class_id < len(class_names):
        return class_names[class_id]
    return str(class_id)


def draw_tracking_results(image, frame_results, class_names=None, track_histories=None, draw_trails=True, trail_length=20, show_state=True):
    if image is None:
        raise ValueError('image must not be None')
    if not frame_results:
        return image

    rendered = image.copy()
    obb_tensor = torch.tensor([result['obb'] for result in frame_results], dtype=torch.float32)
    polygons = xywhr2xyxyxyxy(obb_tensor).cpu().numpy().astype(np.int32)

    for index, result in enumerate(frame_results):
        track_id = result.get('track_id')
        color = track_color(track_id)
        polygon = polygons[index]
        label = f"ID{track_id} {_resolve_label(result.get('class_id', -1), class_names)} {float(result.get('score', 0.0)):.2f}"
        if show_state and result.get('state') not in {None, 'tracked', 'detection'}:
            label = f"{label} {result['state']}"

        cv2.polylines(rendered, [polygon], isClosed=True, color=color, thickness=2, lineType=cv2.LINE_AA)
        origin = tuple(polygon[0])
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0]
        cv2.rectangle(rendered, origin, (origin[0] + text_size[0], origin[1] - text_size[1] - 4), color, -1, cv2.LINE_AA)
        cv2.putText(rendered, label, (origin[0], origin[1] - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        if draw_trails and track_histories is not None and track_id is not None:
            history = track_histories.setdefault(int(track_id), deque(maxlen=int(trail_length)))
            history.append(tuple(int(round(value)) for value in result['obb'][:2]))
            if len(history) >= 2:
                points = np.asarray(history, dtype=np.int32).reshape(-1, 1, 2)
                cv2.polylines(rendered, [points], isClosed=False, color=color, thickness=1, lineType=cv2.LINE_AA)

    return rendered


def render_tracking_sequence(sequence, image_root, output_dir, class_names=None, visualization_cfg=None):
    visualization_cfg = visualization_cfg or {}
    image_root = Path(image_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    track_histories = defaultdict(lambda: deque(maxlen=int(visualization_cfg.get('trail_length', 20))))
    saved_paths = []
    for frame in sequence.get('frames', []):
        image_id = frame.get('image_id')
        if image_id is None:
            continue
        image_path = image_root / image_id
        if not image_path.exists():
            image_path = Path(image_id)
        if not image_path.exists():
            continue

        image = cv2.imread(str(image_path))
        if image is None:
            continue

        rendered = draw_tracking_results(
            image,
            frame.get('results', []),
            class_names=class_names,
            track_histories=track_histories,
            draw_trails=visualization_cfg.get('draw_trails', True),
            trail_length=visualization_cfg.get('trail_length', 20),
            show_state=visualization_cfg.get('show_state', True),
        )
        output_path = output_dir / f"{int(frame.get('frame_index', 0)):06d}.jpg"
        cv2.imwrite(str(output_path), rendered)
        saved_paths.append(str(output_path))
    return saved_paths
