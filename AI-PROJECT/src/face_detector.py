from dataclasses import dataclass

import cv2
import mediapipe as mp
import numpy as np


@dataclass
class FaceDetectionResult:
    faces: list[tuple[int, int, int, int]]
    landmarks_list: list[np.ndarray]

    @property
    def face_count(self):
        return len(self.landmarks_list)

    def primary_landmarks(self):
        if not self.landmarks_list:
            return None
        if len(self.landmarks_list) == 1:
            return self.landmarks_list[0]
        areas = [w * h for _, _, w, h in self.faces]
        return self.landmarks_list[int(np.argmax(areas))]


class FaceDetector:
    def __init__(
        self,
        max_num_faces=5,
        detection_scale=1.0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ):
        self.detection_scale = max(1.0, float(detection_scale))
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=max_num_faces,
            refine_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def detect(self, frame) -> FaceDetectionResult:
        h, w = frame.shape[:2]
        detect_frame = frame
        if self.detection_scale > 1.0:
            detect_frame = cv2.resize(
                frame,
                None,
                fx=self.detection_scale,
                fy=self.detection_scale,
                interpolation=cv2.INTER_LINEAR,
            )
        dh, dw = detect_frame.shape[:2]
        rgb = cv2.cvtColor(detect_frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        faces = []
        landmarks_list = []
        if results.multi_face_landmarks:
            for face_lm in results.multi_face_landmarks:
                pts = np.array(
                    [[lm.x * dw, lm.y * dh, lm.z * dw] for lm in face_lm.landmark],
                    dtype=np.float32,
                )
                if self.detection_scale > 1.0:
                    pts[:, :2] /= self.detection_scale
                    pts[:, 2] /= self.detection_scale
                x, y, box_w, box_h = cv2.boundingRect(pts[:, :2].astype(np.int32))
                x = max(0, min(x, w - 1))
                y = max(0, min(y, h - 1))
                box_w = max(1, min(box_w, w - x))
                box_h = max(1, min(box_h, h - y))
                faces.append((x, y, box_w, box_h))
                landmarks_list.append(pts)

        return FaceDetectionResult(faces, landmarks_list)
