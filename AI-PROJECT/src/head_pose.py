import math

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# 3-D reference model points (generic face, in a right-handed coordinate
# system with X-right, Y-up, Z-out-of-face).  Units are arbitrary but
# internally consistent.  14 landmarks give solvePnP far more constraints
# than the original 6, especially for pitch/roll stability.
# ---------------------------------------------------------------------------
# Indices into the MediaPipe FaceMesh 478-landmark array:
#   1   – nose tip
#   152 – chin
#   33  – left eye outer corner   (subject's left  = image right)
#   263 – right eye outer corner  (subject's right = image left)
#   61  – left mouth corner
#   291 – right mouth corner
#   130 – left eye inner corner
#   359 – right eye inner corner
#   70  – left eyebrow outer
#   300 – right eyebrow outer
#   10  – forehead center (top of face)
#   199 – nose bridge (below nose tip, upper lip area)
#   234 – left ear tragion region
#   454 – right ear tragion region
# ---------------------------------------------------------------------------

LANDMARK_IDS = [
    1, 152, 33, 263, 61, 291,
    130, 359, 70, 300, 10, 199, 234, 454,
]

# Corresponding 3-D coordinates (mm-ish) from an average face model.
# X-right, Y-up, Z-forward (pointing out of the face).
MODEL_POINTS_3D = np.array([
    (   0.0,    0.0,    0.0),    # 1   nose tip
    (   0.0, -330.0,  -65.0),    # 152 chin
    (-225.0,  170.0, -135.0),    # 33  left eye outer
    ( 225.0,  170.0, -135.0),    # 263 right eye outer
    (-150.0, -150.0, -125.0),    # 61  left mouth corner
    ( 150.0, -150.0, -125.0),    # 291 right mouth corner
    (-130.0,  170.0, -125.0),    # 130 left eye inner
    ( 130.0,  170.0, -125.0),    # 359 right eye inner
    (-280.0,  280.0, -150.0),    # 70  left eyebrow outer
    ( 280.0,  280.0, -150.0),    # 300 right eyebrow outer
    (   0.0,  380.0, -100.0),    # 10  forehead center
    (   0.0,  -87.0,   10.0),    # 199 nose bridge / upper lip
    (-400.0,  100.0, -300.0),    # 234 left ear region
    ( 400.0,  100.0, -300.0),    # 454 right ear region
], dtype=np.float64)


def _rotation_matrix_to_euler(R: np.ndarray):
    """Convert a 3×3 rotation matrix to Euler angles (pitch, yaw, roll).

    Uses the ZYX (Tait–Bryan) convention which is natural for head pose:
        R = Rz(roll) · Ry(yaw) · Rx(pitch)

    Returns angles in **degrees** with the following sign convention
    (from the viewer / camera perspective):
        • pitch : positive = looking UP,   negative = looking DOWN
        • yaw   : positive = looking RIGHT (subject turns toward their left)
        • roll  : positive = tilting clockwise (subject's left ear goes up)

    This avoids the ±180° wrapping bug that cv2.RQDecomp3x3 exhibits when
    the head is roughly frontal (pitch near 0 in model-to-camera coords).
    """
    # Extract from R = Rz · Ry · Rx  →  standard aerospace / ZYX formulas
    sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)

    if sy > 1e-6:  # not at gimbal lock
        pitch = math.atan2(R[2, 1], R[2, 2])
        yaw   = math.atan2(-R[2, 0], sy)
        roll  = math.atan2(R[1, 0], R[0, 0])
    else:  # gimbal lock
        pitch = math.atan2(-R[1, 2], R[1, 1])
        yaw   = math.atan2(-R[2, 0], sy)
        roll  = 0.0

    return math.degrees(pitch), math.degrees(yaw), math.degrees(roll)


class HeadPoseEstimator:
    """Estimate head yaw / pitch / roll from MediaPipe FaceMesh landmarks.

    Convention (camera / viewer perspective):
        yaw   : +  → subject looks to their LEFT (camera-right)
        pitch : +  → subject looks UP
        roll  : +  → subject tilts head clockwise (left ear up)

    For a seated person facing the webcam all three values should be
    close to 0 (typically within ±30–40°).
    """

    def estimate(self, frame, landmarks: np.ndarray):
        """Return *(yaw, pitch, roll)* in degrees.

        Parameters
        ----------
        frame : np.ndarray
            BGR image (used only for its dimensions).
        landmarks : np.ndarray, shape (478, 3)
            MediaPipe FaceMesh landmarks in pixel coordinates.

        Returns
        -------
        tuple[float, float, float]
            ``(yaw, pitch, roll)`` in degrees.
        """
        h, w = frame.shape[:2]

        # --- camera intrinsics (approximate) ---
        # A focal length of ~1.2× image width is a reasonable default for a
        # typical webcam with ~60° horizontal FOV.
        focal = w * 1.2
        cx, cy = w / 2.0, h / 2.0
        camera_matrix = np.array([
            [focal,   0.0,  cx],
            [  0.0, focal,  cy],
            [  0.0,   0.0, 1.0],
        ], dtype=np.float64)

        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        # --- gather 2-D image points ---
        image_points = np.array(
            [landmarks[i][:2] for i in LANDMARK_IDS], dtype=np.float64
        )

        # --- solve PnP ---
        success, rvec, tvec = cv2.solvePnP(
            MODEL_POINTS_3D,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not success:
            return 0.0, 0.0, 0.0

        # --- rotation vector → rotation matrix ---
        rmat, _ = cv2.Rodrigues(rvec)

        # --- Euler angles from rotation matrix ---
        # rmat maps model coords to camera coords.  The camera convention is
        # X-right, Y-down, Z-into-screen.  The model is X-right, Y-up,
        # Z-out-of-face.  So the "neutral" rotation (face looking straight at
        # camera) is a 180° rotation about X.
        #
        # We remove that baseline rotation so that a frontal face gives
        # (0, 0, 0) rather than (±180, 0, 0).
        R_baseline = np.array([
            [1,  0,  0],
            [0, -1,  0],
            [0,  0, -1],
        ], dtype=np.float64)
        R_head = rmat @ R_baseline.T   # relative rotation in camera frame

        pitch_deg, yaw_deg, roll_deg = _rotation_matrix_to_euler(R_head)

        # Flip pitch sign so that looking DOWN = negative pitch.
        # The raw ZYX extraction gives +pitch when the face tilts forward
        # (camera-Y direction) which is "looking down" in camera coords,
        # so we negate.
        pitch_deg = -pitch_deg

        return float(yaw_deg), float(pitch_deg), float(roll_deg)


# -----------------------------------------------------------------------
# Quick self-test
# -----------------------------------------------------------------------
if __name__ == "__main__":
    import os
    import sys

    # Resolve paths relative to this file's location
    _this_dir = os.path.dirname(os.path.abspath(__file__))
    _project_root = os.path.dirname(_this_dir)

    ref_img_path = os.path.join(
        _project_root, "database", "reference_images", "202417140.jpg"
    )
    if not os.path.isfile(ref_img_path):
        print(f"ERROR: reference image not found at {ref_img_path}")
        sys.exit(1)

    # Import face detector from sibling module
    sys.path.insert(0, os.path.dirname(_project_root))
    from src.face_detector import FaceDetector

    frame = cv2.imread(ref_img_path)
    if frame is None:
        print(f"ERROR: could not read image {ref_img_path}")
        sys.exit(1)

    detector = FaceDetector(max_num_faces=1)
    result = detector.detect(frame)
    landmarks = result.primary_landmarks()
    if landmarks is None:
        print("ERROR: no face detected in the reference image")
        sys.exit(1)

    estimator = HeadPoseEstimator()
    yaw, pitch, roll = estimator.estimate(frame, landmarks)

    print(f"Reference image : {os.path.basename(ref_img_path)}")
    print(f"Image size      : {frame.shape[1]}×{frame.shape[0]}")
    print(f"Yaw   (+ right) : {yaw:+.1f}°")
    print(f"Pitch (+ up)    : {pitch:+.1f}°")
    print(f"Roll  (+ CW)    : {roll:+.1f}°")
    print()
    ok = all(abs(a) < 45 for a in (yaw, pitch, roll))
    print("[OK] Angles look reasonable for a frontal face." if ok
          else "[!!] WARNING: angles seem too large for a frontal face -- investigate.")
