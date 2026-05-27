from dataclasses import dataclass

import numpy as np


@dataclass
class Track:
    track_id: int
    bbox: tuple[int, int, int, int]
    centroid: np.ndarray
    missed: int = 0


class CentroidTracker:
    """Small CPU-friendly tracker for assigning stable IDs to faces in video."""

    def __init__(self, max_distance=90.0, max_missed=20):
        self.max_distance = max_distance
        self.max_missed = max_missed
        self.next_id = 1
        self.tracks: dict[int, Track] = {}

    @staticmethod
    def _centroid(bbox):
        x, y, w, h = bbox
        return np.array([x + w / 2.0, y + h / 2.0], dtype=np.float32)

    def update(self, bboxes):
        assignments = {}
        centroids = [self._centroid(bbox) for bbox in bboxes]
        unused_detections = set(range(len(bboxes)))
        unused_tracks = set(self.tracks.keys())

        pairs = []
        for track_id, track in self.tracks.items():
            for det_idx, centroid in enumerate(centroids):
                distance = float(np.linalg.norm(track.centroid - centroid))
                pairs.append((distance, track_id, det_idx))

        for distance, track_id, det_idx in sorted(pairs):
            if distance > self.max_distance:
                continue
            if track_id not in unused_tracks or det_idx not in unused_detections:
                continue
            self.tracks[track_id].bbox = bboxes[det_idx]
            self.tracks[track_id].centroid = centroids[det_idx]
            self.tracks[track_id].missed = 0
            assignments[det_idx] = track_id
            unused_tracks.remove(track_id)
            unused_detections.remove(det_idx)

        for det_idx in unused_detections:
            track_id = self.next_id
            self.next_id += 1
            self.tracks[track_id] = Track(track_id, bboxes[det_idx], centroids[det_idx], 0)
            assignments[det_idx] = track_id

        for track_id in list(unused_tracks):
            self.tracks[track_id].missed += 1
            if self.tracks[track_id].missed > self.max_missed:
                del self.tracks[track_id]

        return assignments, self.tracks
