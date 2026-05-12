"""
face_detector.py
Detect faces and extract landmarks with MediaPipe.
"""

import cv2
import mediapipe as mp
import numpy as np


class FaceDetector:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=5,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        # Fallback only. Haar often gives false boxes when the face is tilted.
        self.cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

    def detect(self, frame):
        """
        Returns:
            faces: list[(x, y, w, h)] bounding boxes
            landmarks_list: list[np.ndarray shape (468, 3)] landmark coordinates
        """
        h, w = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        faces = []
        landmarks_list = []

        if results.multi_face_landmarks:
            for face_lm in results.multi_face_landmarks:
                pts = np.array(
                    [[lm.x * w, lm.y * h, lm.z * w] for lm in face_lm.landmark],
                    dtype=np.float32,
                )
                landmarks_list.append(pts)

                x, y, box_w, box_h = cv2.boundingRect(pts[:, :2].astype(np.int32))
                faces.append((x, y, box_w, box_h))
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rects = self.cascade.detectMultiScale(
                gray,
                scaleFactor=1.15,
                minNeighbors=7,
                minSize=(100, 100),
            )
            faces = list(rects) if len(rects) > 0 else []

        return faces, landmarks_list
