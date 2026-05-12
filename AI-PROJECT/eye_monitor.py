"""
eye_monitor.py
Tính EAR (Eye Aspect Ratio) để phát hiện nhắm mắt / ngủ gật
"""

import time
import numpy as np
from scipy.spatial import distance


# Indices landmarks MediaPipe cho mắt trái & phải
# Theo thứ tự: [P1, P2, P3, P4, P5, P6] (EAR formula)
LEFT_EYE  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33,  160, 158, 133, 153, 144]


def _ear(eye_pts: np.ndarray) -> float:
    """Tính Eye Aspect Ratio từ 6 điểm landmarks"""
    A = distance.euclidean(eye_pts[1], eye_pts[5])
    B = distance.euclidean(eye_pts[2], eye_pts[4])
    C = distance.euclidean(eye_pts[0], eye_pts[3])
    return (A + B) / (2.0 * C)


class EyeMonitor:
    def __init__(self, config: dict):
        self.threshold = config["ear_threshold"]
        self.duration  = config["sleep_duration"]
        self._closed_since = None

    def check(self, landmarks: np.ndarray):
        """
        Trả về (ear_value, is_sleeping)
        """
        left_pts  = landmarks[LEFT_EYE][:, :2]
        right_pts = landmarks[RIGHT_EYE][:, :2]

        ear = (_ear(left_pts) + _ear(right_pts)) / 2.0

        now = time.time()

        if ear < self.threshold:
            if self._closed_since is None:
                self._closed_since = now
            elapsed = now - self._closed_since
            sleeping = elapsed >= self.duration
        else:
            self._closed_since = None
            sleeping = False

        return round(ear, 4), sleeping
