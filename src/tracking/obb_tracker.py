from dataclasses import dataclass

import torch

from src.model.bbox_utils import batch_prob_iou


@dataclass
class TrackState:
    track_id: int
    box: torch.Tensor
    score: float
    cls_id: int
    age: int = 0
    hits: int = 1


class OBBTrackletManager:
    """Greedy OBB tracker for video inference."""

    def __init__(self, iou_threshold=0.2, max_age=15, momentum=0.8):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.momentum = momentum
        self.tracks = []
        self.next_track_id = 1

    def _match(self, detections):
        if not self.tracks or detections.numel() == 0:
            return [], list(range(len(self.tracks))), list(range(detections.shape[0]))

        track_boxes = torch.stack([track.box for track in self.tracks], dim=0)
        det_boxes = detections[:, :5]
        iou_matrix = batch_prob_iou(torch.cat([track_boxes, det_boxes], dim=0))
        cross_iou = iou_matrix[:track_boxes.shape[0], track_boxes.shape[0]:]

        matches = []
        used_tracks = set()
        used_dets = set()

        candidates = []
        for track_idx in range(cross_iou.shape[0]):
            for det_idx in range(cross_iou.shape[1]):
                same_class = int(self.tracks[track_idx].cls_id) == int(detections[det_idx, 6].item())
                if same_class:
                    candidates.append((float(cross_iou[track_idx, det_idx]), track_idx, det_idx))

        candidates.sort(key=lambda item: item[0], reverse=True)

        for score, track_idx, det_idx in candidates:
            if score < self.iou_threshold or track_idx in used_tracks or det_idx in used_dets:
                continue
            used_tracks.add(track_idx)
            used_dets.add(det_idx)
            matches.append((track_idx, det_idx))

        unmatched_tracks = [idx for idx in range(len(self.tracks)) if idx not in used_tracks]
        unmatched_dets = [idx for idx in range(detections.shape[0]) if idx not in used_dets]
        return matches, unmatched_tracks, unmatched_dets

    def update(self, detections):
        if detections.numel() == 0:
            for track in self.tracks:
                track.age += 1
            self.tracks = [track for track in self.tracks if track.age <= self.max_age]
            return []

        matches, unmatched_tracks, unmatched_dets = self._match(detections)

        for track_idx, det_idx in matches:
            det = detections[det_idx]
            track = self.tracks[track_idx]
            track.box = self.momentum * track.box + (1.0 - self.momentum) * det[:5]
            track.score = float(det[5].item())
            track.cls_id = int(det[6].item())
            track.age = 0
            track.hits += 1

        for track_idx in unmatched_tracks:
            self.tracks[track_idx].age += 1

        self.tracks = [track for track in self.tracks if track.age <= self.max_age]

        for det_idx in unmatched_dets:
            det = detections[det_idx]
            self.tracks.append(
                TrackState(
                    track_id=self.next_track_id,
                    box=det[:5].clone(),
                    score=float(det[5].item()),
                    cls_id=int(det[6].item()),
                )
            )
            self.next_track_id += 1

        visible_tracks = [track for track in self.tracks if track.age == 0]
        return visible_tracks
