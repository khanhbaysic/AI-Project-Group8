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
    def __init__(self, ear_threshold=0.22, sleep_duration=3.0):
        self.ear_threshold = ear_threshold
        self.sleep_duration = sleep_duration
        self.closed_since = None

    def check(self, landmarks: np.ndarray, now=None):
        if now is None:
            now = time.time()
        left = landmarks[LEFT_EYE][:, :2]
        right = landmarks[RIGHT_EYE][:, :2]
        ear = (eye_aspect_ratio(left) + eye_aspect_ratio(right)) / 2.0

        if ear < self.ear_threshold:
            if self.closed_since is None:
                self.closed_since = now
            sleeping = (now - self.closed_since) >= self.sleep_duration
        else:
            self.closed_since = None
            sleeping = False

        return round(float(ear), 4), sleeping
