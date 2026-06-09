import time

import numpy as np
from scipy.spatial import distance


LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]


def eye_aspect_ratio(eye_pts: np.ndarray) -> float:
    a = distance.euclidean(eye_pts[1], eye_pts[5])
    b = distance.euclidean(eye_pts[2], eye_pts[4])
    c = distance.euclidean(eye_pts[0], eye_pts[3])
    if c <= 1e-6:
        return 0.0
    return (a + b) / (2.0 * c)


class EyeMonitor:
    def __init__(
        self,
        ear_threshold=0.22,
        sleep_duration=3.0,
        use_min_eye_ear=True,
        min_eye_threshold_factor=0.9,
    ):
        self.ear_threshold = ear_threshold
        self.sleep_duration = sleep_duration
        self.use_min_eye_ear = use_min_eye_ear
        self.min_eye_threshold_factor = min_eye_threshold_factor
        self.closed_since = None
        self.last_left_ear = 0.0
        self.last_right_ear = 0.0
        self.last_sleep_ear = 0.0

    def check(self, landmarks: np.ndarray, now=None):
        if now is None:
            now = time.time()
        left = landmarks[LEFT_EYE][:, :2]
        right = landmarks[RIGHT_EYE][:, :2]
        left_ear = eye_aspect_ratio(left)
        right_ear = eye_aspect_ratio(right)
        ear = (left_ear + right_ear) / 2.0
        sleep_ear = min(left_ear, right_ear) if self.use_min_eye_ear else ear
        threshold = (
            self.ear_threshold * self.min_eye_threshold_factor
            if self.use_min_eye_ear
            else self.ear_threshold
        )

        self.last_left_ear = float(left_ear)
        self.last_right_ear = float(right_ear)
        self.last_sleep_ear = float(sleep_ear)

        if sleep_ear < threshold:
            if self.closed_since is None:
                self.closed_since = now
            sleeping = (now - self.closed_since) >= self.sleep_duration
        else:
            self.closed_since = None
            sleeping = False

        return round(float(ear), 4), sleeping
