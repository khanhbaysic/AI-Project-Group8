"""
Quick self-test for SFace identity verification.

Run from the AI-PROJECT folder:
    python src/identity_selftest.py <reference.jpg> <live.jpg> [more_live.jpg ...]

Notes:
  - Image paths are relative to your CURRENT folder (not to this script).
  - The two model files are auto-located in <project>/models/:
        face_detection_yunet_2023mar.onnx
        face_recognition_sface_2021dec.onnx

What to expect:
  - you vs your own photo        -> high cosine, VERIFIED
  - you vs a different person     -> low cosine,  MISMATCH
SFace "same person" threshold (cosine) = 0.363. Lower it (~0.30) if a genuine
match is wrongly rejected; raise it (~0.40) to be stricter.
"""
import sys
from pathlib import Path

import cv2
import numpy as np

THRESHOLD = 0.363
_COS = getattr(cv2, "FaceRecognizerSF_FR_COSINE", 0)
YUNET = "face_detection_yunet_2023mar.onnx"
SFACE = "face_recognition_sface_2021dec.onnx"


def find_models():
    """Look for a 'models' folder next to the script, one level up, or in CWD."""
    here = Path(__file__).resolve()
    candidates = [here.parent, here.parent.parent, Path.cwd()]
    for base in candidates:
        det, rec = base / "models" / YUNET, base / "models" / SFACE
        if det.exists() and rec.exists():
            return det, rec
    searched = "\n  ".join(str(c / "models") for c in candidates)
    raise SystemExit(
        "Could not find the model files. Put BOTH ONNX files in a 'models' "
        f"folder. I looked in:\n  {searched}\nNeeded files: {YUNET}, {SFACE}"
    )


def largest_face(detector, img):
    h, w = img.shape[:2]
    detector.setInputSize((w, h))
    _, faces = detector.detect(img)
    if faces is None or len(faces) == 0:
        return None
    idx = int(np.argmax(faces[:, 2] * faces[:, 3]))
    return faces[idx:idx + 1]


def feature(detector, recognizer, path):
    img = cv2.imread(str(path))
    if img is None:
        raise SystemExit(f"Cannot read image (check the path): {path}")
    face = largest_face(detector, img)
    if face is None:
        raise SystemExit(f"No face found in: {path}")
    return recognizer.feature(recognizer.alignCrop(img, face))


def main():
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)

    det_path, rec_path = find_models()
    detector = cv2.FaceDetectorYN.create(str(det_path), "", (320, 320), 0.7, 0.3, 5000)
    recognizer = cv2.FaceRecognizerSF.create(str(rec_path), "")

    ref = feature(detector, recognizer, sys.argv[1])
    print(f"reference: {sys.argv[1]}  (threshold = {THRESHOLD})\n")
    for live_path in sys.argv[2:]:
        f = feature(detector, recognizer, live_path)
        cosine = float(recognizer.match(ref, f, _COS))
        verdict = ("VERIFIED (same person)" if cosine >= THRESHOLD
                   else "MISMATCH (different person)")
        print(f"  {live_path:40s} cosine={cosine:+.3f}  ->  {verdict}")


if __name__ == "__main__":
    main()