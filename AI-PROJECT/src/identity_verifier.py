from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


SHAPE_LANDMARKS = [
    10, 152, 234, 454, 127, 356,
    33, 133, 362, 263,
    61, 291, 13, 14,
    1, 2, 4, 5, 98, 327,
    168, 197, 195, 199,
]


@dataclass
class IdentityResult:
    status: str
    similarity: float
    message: str = ""


class IdentityVerifier:
    """Pluggable identity verification.

    Current implementation uses a lightweight FaceMesh landmark embedding and
    cosine similarity. For a production-grade system, replace this module with
    ArcFace/InsightFace/DeepFace embeddings without changing main.py.
    """

    def __init__(self, face_detector, threshold=0.88):
        self.face_detector = face_detector
        self.threshold = threshold
        self.reference_embedding = None
        self.reference_path = None

    def load_reference(self, image_path: Path):
        image_path = Path(image_path)
        if not image_path.exists():
            return False, f"Missing reference image: {image_path}"
        image = cv2.imread(str(image_path))
        if image is None:
            return False, f"Could not read reference image: {image_path}"

        detection = self.face_detector.detect(image)
        landmarks = detection.primary_landmarks()
        if landmarks is None:
            return False, f"No face found in reference image: {image_path}"

        embedding = self.extract_embedding(landmarks)
        if embedding is None:
            return False, "Failed embedding extraction from reference image"

        self.reference_embedding = embedding
        self.reference_path = image_path
        return True, "Reference loaded"

    def extract_embedding(self, landmarks):
        points = landmarks[:, :2]
        face_width = np.linalg.norm(points[234] - points[454])
        if face_width <= 1e-6:
            return None

        left_eye = points[33]
        right_eye = points[263]
        eye_center = (left_eye + right_eye) / 2.0
        angle = -np.arctan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0])
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        rotation = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)

        normalized = (points[SHAPE_LANDMARKS] - eye_center) / face_width
        aligned = normalized @ rotation.T
        embedding = aligned.flatten().astype(np.float32)
        norm = np.linalg.norm(embedding)
        if norm <= 1e-6:
            return None
        return embedding / norm

    def verify(self, live_landmarks):
        if self.reference_embedding is None:
            return IdentityResult("UNKNOWN", 0.0, "No reference identity loaded")

        live_embedding = self.extract_embedding(live_landmarks)
        if live_embedding is None:
            return IdentityResult("ERROR", 0.0, "Failed embedding extraction from live face")

        cosine = float(np.dot(self.reference_embedding, live_embedding))
        similarity = max(0.0, min(1.0, (cosine + 1.0) / 2.0))
        status = "VERIFIED" if similarity >= self.threshold else "MISMATCH"
        return IdentityResult(status, round(similarity, 3))
