import time
from collections import deque

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
    """Detect talking from MAR oscillation rather than sustained mouth opening.

    Speech produces rapid open/close jaw cycles.  Over a short rolling window
    (default 1.5 s) we count the number of times MAR crosses the *open*
    threshold in the upward direction (rising-edge transitions) and also
    measure the variance of the MAR signal.  If either metric exceeds its
    configured threshold the frame is considered "talking".

    A final *confirmation duration* (``talk_duration``) requires the
    oscillation signal to persist for a minimum time before the talking flag
    is raised, filtering out isolated one-off mouth movements.

    Why this is better than "MAR > threshold for N seconds":
    * A yawn holds the mouth open steadily — high MAR but low variance and
      zero transitions → **not** flagged.
    * Real speech rapidly alternates open/close — moderate MAR with high
      variance and many transitions → **flagged**.

    Constructor parameters
    ----------------------
    mar_threshold : float
        MAR value above which the mouth is considered "open" for the purpose
        of counting open/close transitions.  (Same role as the old threshold,
        kept as the first positional arg for backward compatibility.)
    talk_duration : float
        Seconds of sustained oscillation required before the flag turns True.
        Kept as the second positional arg for backward compatibility.
    talk_window : float
        Length (seconds) of the rolling window for oscillation analysis.
    talk_min_transitions : int
        Minimum upward (closed→open) transitions within the window to
        consider the signal as speech.
    talk_mar_variance_threshold : float
        Minimum MAR variance within the window to consider the signal as
        speech.  Acts as a secondary / alternative trigger alongside
        transition count.
    """

    def __init__(
        self,
        mar_threshold: float = 0.60,
        talk_duration: float = 2.0,
        talk_window: float = 1.5,
        talk_min_transitions: int = 3,
        talk_mar_variance_threshold: float = 0.005,
    ):
        self.mar_threshold = mar_threshold
        self.talk_duration = talk_duration
        self.talk_window = talk_window
        self.talk_min_transitions = talk_min_transitions
        self.talk_mar_variance_threshold = talk_mar_variance_threshold

        # Rolling buffer of (timestamp, mar_value) pairs
        self._history: deque[tuple[float, float]] = deque()
        # Track the previous "open" state for edge detection
        self._prev_open: bool = False
        # Timestamp when oscillation was first detected continuously
        self._oscillation_since: float | None = None

    def check(self, landmarks: np.ndarray, now=None):
        """Return ``(mar, talking_bool)`` — signature unchanged.

        Parameters
        ----------
        landmarks : np.ndarray, shape (478, 3)
            MediaPipe FaceMesh landmarks in pixel coordinates.
        now : float or None
            Current timestamp (seconds).  Defaults to ``time.time()``.
        """
        if now is None:
            now = time.time()

        pts = landmarks[MOUTH_POINTS][:, :2]
        mar = mouth_aspect_ratio(pts)

        # --- maintain rolling window ---
        self._history.append((now, mar))
        cutoff = now - self.talk_window
        while self._history and self._history[0][0] < cutoff:
            self._history.popleft()

        # --- count rising-edge transitions (closed → open) ---
        is_open = mar > self.mar_threshold
        transitions = 0
        prev = self._prev_open
        mars = []
        for _, m in self._history:
            o = m > self.mar_threshold
            if o and not prev:
                transitions += 1
            prev = o
            mars.append(m)
        self._prev_open = is_open

        # --- MAR variance over the window ---
        mar_var = float(np.var(mars)) if len(mars) >= 2 else 0.0

        # --- decide if oscillation is happening right now ---
        oscillating = (
            transitions >= self.talk_min_transitions
            or mar_var >= self.talk_mar_variance_threshold
        )

        # --- require oscillation to persist for talk_duration ---
        if oscillating:
            if self._oscillation_since is None:
                self._oscillation_since = now
            talking = (now - self._oscillation_since) >= self.talk_duration
        else:
            self._oscillation_since = None
            talking = False

        return round(float(mar), 4), talking
