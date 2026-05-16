from collections import deque

import numpy as np


class LivenessDetector:
    """CPU-friendly liveness estimate using blink and motion cues.

    This is not a replacement for a trained anti-spoofing model. It is a
    lightweight, explainable detector suitable for an academic prototype.
    """

    def __init__(self, threshold=0.35, warmup_seconds=4.0, window_seconds=8.0, ear_threshold=0.22):
        self.threshold = threshold
        self.warmup_seconds = warmup_seconds
        self.window_seconds = window_seconds
        self.ear_threshold = ear_threshold
        self.samples = deque()
        self.started_at = None
        self._was_open = True
        self._blink_count = 0

    def update(self, now, landmarks, ear, yaw, pitch):
        if self.started_at is None:
            self.started_at = now

        center = np.mean(landmarks[:, :2], axis=0)
        face_width = np.linalg.norm(landmarks[234][:2] - landmarks[454][:2])
        normalized_center = center / max(face_width, 1e-6)

        is_open = ear >= self.ear_threshold
        if self._was_open and not is_open:
            self._blink_count += 1
        self._was_open = is_open

        self.samples.append({
            "time": now,
            "ear": float(ear),
            "yaw": float(yaw),
            "pitch": float(pitch),
            "center": normalized_center,
            "blink_count": self._blink_count,
        })
        while self.samples and now - self.samples[0]["time"] > self.window_seconds:
            self.samples.popleft()

        elapsed = now - self.started_at
        if len(self.samples) < 2:
            return 0.5, "CHECKING"

        first = self.samples[0]
        last = self.samples[-1]
        blink_delta = last["blink_count"] - first["blink_count"]
        centers = np.array([s["center"] for s in self.samples], dtype=np.float32)
        yaws = np.array([s["yaw"] for s in self.samples], dtype=np.float32)
        pitches = np.array([s["pitch"] for s in self.samples], dtype=np.float32)

        blink_score = min(1.0, blink_delta / 1.0)
        face_motion_score = min(1.0, float(np.std(centers, axis=0).mean()) * 8.0)
        head_motion_score = min(1.0, (float(np.std(yaws)) + float(np.std(pitches))) / 8.0)

        score = 0.45 * blink_score + 0.30 * face_motion_score + 0.25 * head_motion_score
        if elapsed < self.warmup_seconds:
            return max(0.5, score), "CHECKING"

        status = "LIVE" if score >= self.threshold else "SPOOFING"
        return round(float(score), 3), status
