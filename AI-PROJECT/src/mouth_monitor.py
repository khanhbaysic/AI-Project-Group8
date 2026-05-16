import time

import numpy as np
from scipy.spatial import distance


MOUTH_POINTS = [61, 291, 39, 269, 0, 17]


def mouth_aspect_ratio(mouth_pts: np.ndarray) -> float:
    a = distance.euclidean(mouth_pts[2], mouth_pts[5])
    b = distance.euclidean(mouth_pts[3], mouth_pts[4])
    c = distance.euclidean(mouth_pts[0], mouth_pts[1])
    if c <= 1e-6:
        return 0.0
    return (a + b) / (2.0 * c)


class MouthMonitor:
    def __init__(self, mar_threshold=0.6, talk_duration=2.0):
        self.mar_threshold = mar_threshold
        self.talk_duration = talk_duration
        self.open_since = None

    def check(self, landmarks: np.ndarray, now=None):
        if now is None:
            now = time.time()
        pts = landmarks[MOUTH_POINTS][:, :2]
        mar = mouth_aspect_ratio(pts)

        if mar > self.mar_threshold:
            if self.open_since is None:
                self.open_since = now
            talking = (now - self.open_since) >= self.talk_duration
        else:
            self.open_since = None
            talking = False

        return round(float(mar), 4), talking
