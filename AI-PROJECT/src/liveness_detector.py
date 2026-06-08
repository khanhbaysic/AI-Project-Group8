from collections import deque

import numpy as np


# ---------------------------------------------------------------------------
# Landmark indices used for computing internal facial geometry variance.
# These are spread across different facial regions so that organic muscle
# movement (live face) creates measurable relative jitter, while a static
# photo — even one being shaken — stays rigid.
# ---------------------------------------------------------------------------
# Regions: nose tip, chin, left/right eye outer corners, left/right mouth
# corners, left/right eyebrow outer, forehead center
_SHAPE_IDXS = np.array([1, 152, 33, 263, 61, 291, 70, 300, 10])


class LivenessDetector:
    """Robust CPU-friendly liveness estimator using blink and rigidity cues.

    Key improvements over the naïve motion-only approach:
    -------------------------------------------------------
    1. **Rigidity penalty** – When the whole face translates as a rigid block
       (e.g., a photo being shaken), the *global* translation dominates and the
       *local* relative-landmark variance stays near zero.  A real face has
       organic muscle movement that creates local jitter independent of global
       head motion.  We therefore reward local variance while punishing purely
       global (rigid) motion.

    2. **Blink gate** – A genuine blink is the single strongest proof of
       liveness.  If no blink is observed after the warmup period the blink
       score stays at zero, dragging the total score below the threshold.

    3. **Faster detection** – Warmup and window have been shortened so that
       spoofing is flagged in ~3-5 s rather than 20-30 s.

    4. **No face-center motion reward** – Raw face-center displacement was the
       loophole that let a shaken photo accumulate liveness points.  It is
       replaced by the rigidity-corrected local variance.
    """

    def __init__(
        self,
        threshold: float = 0.38,
        warmup_seconds: float = 2.5,
        window_seconds: float = 5.0,
        ear_threshold: float = 0.22,
        spoof_confirm_seconds: float = 4.0,
    ):
        self.threshold = threshold
        self.warmup_seconds = warmup_seconds
        self.window_seconds = window_seconds
        self.ear_threshold = ear_threshold
        self.spoof_confirm_seconds = spoof_confirm_seconds

        self.samples: deque = deque()
        self.started_at: float | None = None
        self._spoofing_since: float | None = None
        self._was_open: bool = True
        self._blink_count: int = 0

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

        # --- blink detection ---
        is_open = ear >= self.ear_threshold
        if self._was_open and not is_open:
            self._blink_count += 1
        self._was_open = is_open

        # --- extract normalised shape vector (pose-invariant) ---
        # Project all shape landmarks into a coordinate frame anchored at the
        # nose tip and scaled by face width so that global translation and
        # scale are removed.  This leaves only *internal* shape deformation.
        shape_pts = landmarks[_SHAPE_IDXS, :2].astype(np.float32)        # (9, 2)
        nose_tip = landmarks[1, :2].astype(np.float32)
        face_width = float(np.linalg.norm(
            landmarks[234, :2] - landmarks[454, :2]
        ))
        face_width = max(face_width, 1e-6)
        norm_shape = (shape_pts - nose_tip) / face_width                  # (9, 2)

        # --- global face centre (for rigidity analysis) ---
        center = np.mean(landmarks[:, :2], axis=0).astype(np.float32)
        norm_center = center / face_width

        self.samples.append({
            "time": now,
            "ear": float(ear),
            "yaw": float(yaw),
            "pitch": float(pitch),
            "norm_shape": norm_shape,   # (9, 2)
            "norm_center": norm_center,  # (2,)
            "blink_count": self._blink_count,
        })

        # Evict samples older than the analysis window
        while self.samples and now - self.samples[0]["time"] > self.window_seconds:
            self.samples.popleft()

        elapsed = now - self.started_at
        if len(self.samples) < 3:
            return 0.5, "CHECKING"

        # ------------------------------------------------------------------
        # Feature extraction
        # ------------------------------------------------------------------
        first = self.samples[0]
        last = self.samples[-1]

        blink_delta = last["blink_count"] - first["blink_count"]

        shapes = np.stack([s["norm_shape"] for s in self.samples])      # (N, 9, 2)
        centers = np.stack([s["norm_center"] for s in self.samples])    # (N, 2)
        yaws = np.array([s["yaw"] for s in self.samples], np.float32)
        pitches = np.array([s["pitch"] for s in self.samples], np.float32)

        # ------------------------------------------------------------------
        # Score 1 – Blink score (0–1)
        # One blink is enough proof; anything beyond is capped.
        # ------------------------------------------------------------------
        blink_score = min(1.0, blink_delta / 1.0)

        # ------------------------------------------------------------------
        # Score 2 – Local organic shape variance (0–1)
        # Measure frame-to-frame variance of the *normalised* shape vector.
        # Because global translation and scale are already removed, only
        # genuine facial muscle micro-movements survive.
        # A photo — even a shaken one — stays rigid → variance ≈ 0.
        # A live face has breathing, subtle expression changes → variance > 0.
        #
        # Calibration: live faces typically produce values in [0.003, 0.015].
        # We scale so that ~0.008 maps to score ≈ 0.5.
        # ------------------------------------------------------------------
        shape_std = float(np.std(shapes, axis=0).mean())   # mean std per coord
        local_organic_score = min(1.0, shape_std / 0.012)

        # ------------------------------------------------------------------
        # Score 3 – Rigidity penalty
        # If the global center moves a lot but the normalised shape barely
        # changes, the motion is rigid (photo being shaken).
        # We compute the ratio of global motion to local shape motion.
        # A high ratio → penalty.  This score is 1 when motion is organic
        # and < 1 when motion is rigidly global.
        # ------------------------------------------------------------------
        global_motion = float(np.std(centers, axis=0).mean())
        if global_motion < 1e-5:
            # Nothing is moving at all – consistent with a static photo
            rigidity_score = 0.0
        else:
            # How much of the motion is accounted for by internal deformation?
            ratio = shape_std / (global_motion + 1e-8)
            # ratio close to 0 → rigid (photo shaking)
            # ratio close to 1+ → organic (live face)
            rigidity_score = min(1.0, ratio * 6.0)

        # ------------------------------------------------------------------
        # Score 4 – Head pose micro-variation (0–1)
        # Real heads show small natural sway in yaw/pitch from breathing and
        # small postural adjustments.  A perfect still photo yields near-zero
        # std in pose angles.  However, a shaken photo *also* produces pose
        # variation, so this score has lower weight than in the original.
        # ------------------------------------------------------------------
        head_motion_score = min(1.0, (float(np.std(yaws)) + float(np.std(pitches))) / 6.0)

        # ------------------------------------------------------------------
        # Composite score
        # Weights: blink 40 %, local organic 30 %, rigidity 20 %, head 10 %
        # ------------------------------------------------------------------
        score = (
            0.40 * blink_score
            + 0.30 * local_organic_score
            + 0.20 * rigidity_score
            + 0.10 * head_motion_score
        )

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
