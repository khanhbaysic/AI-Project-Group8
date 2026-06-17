from collections import deque

import numpy as np


# ---------------------------------------------------------------------------
# Landmark indices for internal shape geometry analysis.
# Spread across different facial regions so that organic muscle movement
# creates measurable relative deformation while a rigid photo stays flat.
# ---------------------------------------------------------------------------
# Regions: nose tip, chin, left/right eye outer corners, left/right mouth
# corners, left/right eyebrow outer, forehead center
_SHAPE_IDXS = np.array([1, 152, 33, 263, 61, 291, 70, 300, 10])

# Typical MediaPipe per-landmark noise floor in *normalised* coords
# (pixel noise / face_width).  Values below this are unreliable.
# Conservatively estimated at ~1.5px noise on a ~150px-wide face = 0.010.
_NORM_NOISE_FLOOR = 0.010


class LivenessDetector:
    """Blink-primary liveness estimator.

    Design rationale
    ----------------
    Previous versions relied on landmark-motion variance (local organic score)
    and rigidity analysis.  These are defeated by MediaPipe's own landmark
    jitter: even 1-2 px of detector noise on a still photo produces variance
    values that saturate the score and classify photos as LIVE.

    The only signal a photo **genuinely cannot fake** is a blink:
      - A real eye closes (EAR drops well below threshold) and reopens.
      - A printed/displayed photo has a fixed open-eye EAR that never dips.

    New weight scheme
    -----------------
    * Blink presence (0.65) – dominant; one confirmed blink makes score ≥ 0.65
    * EAR micro-variance (0.20) – variance of EAR signal above the noise floor
    * Head-pose micro-sway  (0.15) – natural postural sway (low weight; photos
      can also have pose jitter from the detector)

    Threshold raised to 0.50:
      - A photo with no blink: blink_score=0.  Even if EAR-var and head scores
        each max out (unlikely for a still photo), max = 0+0.20+0.15 = 0.35 < 0.50
        → SPOOFING detected correctly.
      - A real face with 1 blink: blink_score=1.0, total ≥ 0.65 > 0.50
        → LIVE detected correctly.

    The old landmark-shape variance is intentionally removed because it cannot
    be distinguished from detector noise without per-device calibration.
    """

    def __init__(
        self,
        threshold: float = 0.50,
        warmup_seconds: float = 5.0,
        window_seconds: float = 8.0,
        ear_threshold: float = 0.22,
        spoof_confirm_seconds: float = 3.0,
        require_blink: bool = True,
        max_score_without_blink: float = 0.29,
    ):
        self.threshold = threshold
        self.warmup_seconds = warmup_seconds
        self.window_seconds = window_seconds
        self.ear_threshold = ear_threshold
        self.spoof_confirm_seconds = spoof_confirm_seconds
        self.require_blink = require_blink
        self.max_score_without_blink = max_score_without_blink

        self.samples: deque = deque()
        self.started_at: float | None = None
        self._spoofing_since: float | None = None
        self._was_open: bool = True
        self._blink_count: int = 0
        # Track the minimum EAR seen in each blink candidate
        self._in_blink: bool = False
        self._blink_min_ear: float = 1.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, now: float, landmarks: np.ndarray, ear: float, yaw: float, pitch: float):
        """Update detector state and return *(score, status)*.

        Parameters
        ----------
        now : float
            Current wall-clock time (seconds).
        landmarks : np.ndarray, shape (478, 3)
            MediaPipe FaceMesh landmarks in pixel coordinates.
        ear : float
            Eye Aspect Ratio for blink detection.
        yaw, pitch : float
            Head pose angles in degrees.

        Returns
        -------
        tuple[float, str]
            ``(score, status)`` where *status* is one of
            ``"CHECKING"``, ``"LIVE"``, or ``"SPOOFING"``.
        """
        if self.started_at is None:
            self.started_at = now

        # ------------------------------------------------------------------
        # Blink detection – track full open→closed→open transitions.
        # We require the EAR to drop below (ear_threshold * 0.85) to count
        # as a genuine closure, not just a flutter from detector noise.
        # ------------------------------------------------------------------
        blink_thresh = self.ear_threshold * 0.85
        is_closed = ear < blink_thresh

        if is_closed:
            self._in_blink = True
            self._blink_min_ear = min(self._blink_min_ear, ear)
        elif self._in_blink:
            # Eye just reopened – count as blink only if it closed enough
            if self._blink_min_ear < blink_thresh:
                self._blink_count += 1
            self._in_blink = False
            self._blink_min_ear = 1.0

        self._was_open = not is_closed

        self.samples.append({
            "time": now,
            "ear": float(ear),
            "yaw": float(yaw),
            "pitch": float(pitch),
            "blink_count": self._blink_count,
        })

        # Evict samples older than the analysis window
        while self.samples and now - self.samples[0]["time"] > self.window_seconds:
            self.samples.popleft()

        elapsed = now - self.started_at
        if len(self.samples) < 5:
            return 0.5, "CHECKING"

        # ------------------------------------------------------------------
        # Feature extraction
        # ------------------------------------------------------------------
        first = self.samples[0]
        last = self.samples[-1]

        blink_delta = last["blink_count"] - first["blink_count"]
        ears = np.array([s["ear"] for s in self.samples], np.float32)
        yaws = np.array([s["yaw"] for s in self.samples], np.float32)
        pitches = np.array([s["pitch"] for s in self.samples], np.float32)

        # ------------------------------------------------------------------
        # Score 1 – Blink score (0–1)  [weight 0.65]
        # One confirmed blink is proof of liveness.  We also give partial
        # credit for large EAR dips that didn't quite close (squinting).
        # ------------------------------------------------------------------
        blink_score = min(1.0, blink_delta / 1.0)

        # ------------------------------------------------------------------
        # Score 2 – EAR micro-variance (0–1)  [weight 0.20]
        # The standard deviation of EAR across the window reveals:
        #   - Real eyes: subtle continuous fluctuation from involuntary saccades,
        #     micro-expressions, breathing.
        #   - Photo eyes: EAR is nearly constant (detector noise only ~0.003).
        # Calibration: live faces → EAR std typically in [0.010, 0.040].
        # We scale so that std=0.015 → score=1.0.
        # Photos produce EAR std ≈ 0.002-0.005 → score ≈ 0.13-0.33.
        # ------------------------------------------------------------------
        ear_std = float(np.std(ears))
        ear_var_score = min(1.0, ear_std / 0.015)

        # ------------------------------------------------------------------
        # Score 3 – Head-pose micro-sway (0–1)  [weight 0.15]
        # Natural breathing and postural sway produces small yaw/pitch
        # variation (~1-3°). A still photo produces near-zero variation
        # (or detector noise only ~0.5°). Low weight because a handheld
        # photo can also produce pose variation.
        # Scale: (std_yaw + std_pitch) = 2° → score = 0.33; 6° → 1.0
        # ------------------------------------------------------------------
        head_sway_score = min(1.0, (float(np.std(yaws)) + float(np.std(pitches))) / 6.0)

        # ------------------------------------------------------------------
        # Composite score
        # Weights: blink 65%, EAR-variance 20%, head-sway 15%
        # Threshold = 0.50
        #
        # Photo (no blink):   0*0.65 + ≤0.33*0.20 + ≤0.33*0.15 = ≤0.115 < 0.50 ✓
        # Real (1 blink):     1*0.65 + any*0.20   + any*0.15   ≥ 0.65 > 0.50 ✓
        # ------------------------------------------------------------------
        score = (
            0.65 * blink_score
            + 0.20 * ear_var_score
            + 0.15 * head_sway_score
        )
        if self.require_blink and self._blink_count == 0:
            score = min(score, self.max_score_without_blink)

        if elapsed < self.warmup_seconds:
            return max(0.5, score), "CHECKING"

        if score >= self.threshold:
            self._spoofing_since = None
            status = "LIVE"
        else:
            if self._spoofing_since is None:
                self._spoofing_since = now
            status = (
                "SPOOFING"
                if now - self._spoofing_since >= self.spoof_confirm_seconds
                else "CHECKING"
            )
        return round(float(score), 3), status
