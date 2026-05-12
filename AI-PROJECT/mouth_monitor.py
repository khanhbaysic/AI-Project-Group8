"""
mouth_monitor.py
Tính MAR (Mouth Aspect Ratio) để phát hiện nói chuyện
"""

import time
import numpy as np
from scipy.spatial import distance


# Indices landmarks MediaPipe cho miệng
MOUTH_OUTER = [61, 291, 39, 269, 0, 17]
# [left, right, top-left, top-right, top-center, bottom-center]


def _mar(mouth_pts: np.ndarray) -> float:
    """Tính Mouth Aspect Ratio"""
    A = distance.euclidean(mouth_pts[2], mouth_pts[5])  # dọc trái
    B = distance.euclidean(mouth_pts[3], mouth_pts[4])  # dọc phải — dùng top/bottom center
    C = distance.euclidean(mouth_pts[0], mouth_pts[1])  # ngang
    return (A + B) / (2.0 * C)


class MouthMonitor:
    def __init__(self, config: dict):
        self.threshold = config["mar_threshold"]
        self.duration  = config["talk_duration"]
        self._open_since = None

    def check(self, landmarks: np.ndarray):
        """
        Trả về (mar_value, is_talking)
        """
        mouth_pts = landmarks[MOUTH_OUTER][:, :2]
        mar = _mar(mouth_pts)

        now = time.time()

        if mar > self.threshold:
            if self._open_since is None:
                self._open_since = now
            elapsed = now - self._open_since
            talking = elapsed >= self.duration
        else:
            self._open_since = None
            talking = False

        return round(mar, 4), talking
