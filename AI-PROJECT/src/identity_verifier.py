from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


# SFace returns a similarity value; for the COSINE distance type a LARGER value
# means MORE similar. OpenCV's reference threshold for "same person" is 0.363.
_FR_COSINE = getattr(cv2, "FaceRecognizerSF_FR_COSINE", 0)


@dataclass
class IdentityResult:
    status: str
    similarity: float
    message: str = ""


class IdentityVerifier:
    """Face-recognition based identity verification (OpenCV SFace + YuNet).

    Why this replaces the old version
    ---------------------------------
    The previous implementation built an "embedding" from MediaPipe FaceMesh
    landmark *geometry* (face outline + eye/mouth/nose positions), normalised
    and compared with cosine similarity. That cannot tell two people apart:
    every human face has the same mesh topology and roughly the same
    proportions, so after centering + scaling the vectors are almost identical
    for everyone (cosine ~ 0.99). On top of that the score was remapped with
    (cosine + 1) / 2, which pushes everything into [0.5, 1.0]. The result is
    that essentially any face scored above the 0.88 threshold and was reported
    as VERIFIED -- including the wrong person.

    This version uses a real face-recognition model. YuNet detects and aligns
    the face; SFace produces a 128-D *appearance* embedding trained so that the
    same person scores high and different people score low, with a clear
    separating threshold (cosine >= 0.363 => same person).

    Fail-safe by design: if the models are missing, no face is found, or
    anything goes wrong, it returns UNKNOWN / NO_FACE / ERROR -- never VERIFIED.
    """

    def __init__(self, detector_model, recognizer_model, threshold=0.363,
                 check_interval=0.5, det_score=0.7):
        self.threshold = float(threshold)
        self.check_interval = float(check_interval)
        self.reference_feature = None
        self.reference_path = None
        self._last_time = None
        self._last_result = None

        self.ready = False
        self.init_error = ""
        try:
            self.detector = cv2.FaceDetectorYN.create(
                str(detector_model), "", (320, 320), det_score, 0.3, 5000
            )
            self.recognizer = cv2.FaceRecognizerSF.create(str(recognizer_model), "")
            self.ready = True
        except Exception as exc:  # missing/invalid model files, etc.
            self.detector = None
            self.recognizer = None
            self.init_error = (
                f"Could not load face-recognition models "
                f"('{detector_model}', '{recognizer_model}'): {exc}"
            )

    # ------------------------------------------------------------------ #
    # internals
    # ------------------------------------------------------------------ #
    def _largest_face(self, image):
        """Return the largest detected face row (shape (1, 15)) or None."""
        h, w = image.shape[:2]
        self.detector.setInputSize((w, h))
        _, faces = self.detector.detect(image)
        if faces is None or len(faces) == 0:
            return None
        # columns 2,3 are width,height of each detection -> pick biggest face
        idx = int(np.argmax(faces[:, 2] * faces[:, 3]))
        return faces[idx:idx + 1]

    def _feature(self, image, face_row):
        aligned = self.recognizer.alignCrop(image, face_row)
        return self.recognizer.feature(aligned)

    # ------------------------------------------------------------------ #
    # public API (same names/return type as before)
    # ------------------------------------------------------------------ #
    def load_reference(self, image_path):
        if not self.ready:
            return False, self.init_error
        image_path = Path(image_path)
        if not image_path.exists():
            return False, f"Missing reference image: {image_path}"
        image = cv2.imread(str(image_path))
        if image is None:
            return False, f"Could not read reference image: {image_path}"

        face = self._largest_face(image)
        if face is None:
            return False, f"No face found in reference image: {image_path}"

        self.reference_feature = self._feature(image, face)
        self.reference_path = image_path
        self._last_time = None
        self._last_result = None
        return True, "Reference loaded"

    def verify(self, frame, now=None):
        """Compare the live frame against the loaded reference.

        `frame` is the live BGR image (NOT FaceMesh landmarks). `now` is an
        optional timestamp; when provided, the result is cached and only
        recomputed every `check_interval` seconds to save CPU.
        """
        # Throttle: reuse the last verdict within the check interval.
        if (now is not None and self._last_time is not None
                and self._last_result is not None
                and (now - self._last_time) < self.check_interval):
            return self._last_result

        if not self.ready:
            result = IdentityResult("ERROR", 0.0, self.init_error)
        elif self.reference_feature is None:
            result = IdentityResult("UNKNOWN", 0.0, "No reference identity loaded")
        else:
            face = self._largest_face(frame)
            if face is None:
                result = IdentityResult("NO_FACE", 0.0, "No face detected in frame")
            else:
                feat = self._feature(frame, face)
                cosine = float(self.recognizer.match(
                    self.reference_feature, feat, _FR_COSINE))
                status = "VERIFIED" if cosine >= self.threshold else "MISMATCH"
                result = IdentityResult(status, round(cosine, 3))

        if now is not None:
            self._last_time = now
            self._last_result = result
        return result