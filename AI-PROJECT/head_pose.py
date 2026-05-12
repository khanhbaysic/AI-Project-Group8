"""
head_pose.py
Estimate head yaw / pitch / roll with cv2.solvePnP.
"""

import cv2
import numpy as np


# Standard 3D face model points in millimeters.
MODEL_POINTS_3D = np.array([
    (0.0, 0.0, 0.0),          # Nose tip
    (0.0, -330.0, -65.0),     # Chin
    (-225.0, 170.0, -135.0),  # Left eye outer corner
    (225.0, 170.0, -135.0),   # Right eye outer corner
    (-150.0, -150.0, -125.0), # Left mouth corner
    (150.0, -150.0, -125.0),  # Right mouth corner
], dtype=np.float64)

# MediaPipe landmark ids in the same order as MODEL_POINTS_3D.
LANDMARK_IDS = [1, 152, 33, 263, 61, 291]


class HeadPoseEstimator:
    def estimate(self, frame, landmarks: np.ndarray):
        """
        landmarks: np.ndarray shape (468, 3), pixel coordinates.

        Returns (yaw, pitch, roll) in degrees:
            yaw   > 0 -> turn right, < 0 -> turn left
            pitch > 0 -> look down,  < 0 -> look up
            roll  -> head tilt
        """
        h, w = frame.shape[:2]

        focal = w
        center = (w / 2, h / 2)
        camera_matrix = np.array([
            [focal, 0, center[0]],
            [0, focal, center[1]],
            [0, 0, 1],
        ], dtype=np.float64)

        dist_coeffs = np.zeros((4, 1))
        image_points = np.array(
            [landmarks[i][:2] for i in LANDMARK_IDS],
            dtype=np.float64,
        )

        success, rvec, tvec = cv2.solvePnP(
            MODEL_POINTS_3D,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        if not success:
            return 0.0, 0.0, 0.0

        rmat, _ = cv2.Rodrigues(rvec)
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

        pitch = angles[0]
        yaw = angles[1]
        roll = angles[2]

        return yaw, pitch, roll
