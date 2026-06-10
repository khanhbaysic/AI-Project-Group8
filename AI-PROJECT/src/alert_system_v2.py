import csv
import time
from datetime import datetime
from pathlib import Path

import cv2


class AlertSystem:
    """Record violations + save evidence.

    NEW: privacy mode. When privacy_mode=True, the evidence image has its
    face region BLURRED (or the whole frame if no bbox) before it is
    saved -> reducing the risk of leaking biometric data. privacy_mode
    defaults to False to KEEP the old behaviour (nothing running breaks).

    privacy_blur_strength: blur amount (odd; larger = blurrier). 35 is fine.
    """

    def __init__(self, csv_path: Path, image_dir: Path, cooldown_seconds=5.0,
                 privacy_mode=False, privacy_blur_strength=35):
        self.csv_path = Path(csv_path)
        self.image_dir = Path(image_dir)
        self.cooldown_seconds = cooldown_seconds
        self.privacy_mode = privacy_mode
        self.privacy_blur_strength = privacy_blur_strength
        self.last_trigger = {}
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.image_dir.mkdir(parents=True, exist_ok=True)
        if not self.csv_path.exists():
            with self.csv_path.open("w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp", "student_id", "alert_type", "state",
                    "attention_score", "liveness_score", "face_similarity",
                    "ear", "mar", "yaw", "pitch", "image_path",
                ])

    def _blur_face(self, image, bbox):
        """Blur the face region. bbox = (x, y, w, h) or None.

        If there is no bbox -> blur the whole frame (privacy-safe).
        Returns a new image; does NOT modify the original.
        """
        out = image.copy()
        k = self.privacy_blur_strength
        if k % 2 == 0:
            k += 1  # gaussian kernel must be odd

        if bbox is not None:
            x, y, w, h = (int(v) for v in bbox)
            x = max(0, x); y = max(0, y)
            x2 = min(out.shape[1], x + w)
            y2 = min(out.shape[0], y + h)
            if x2 > x and y2 > y:
                roi = out[y:y2, x:x2]
                # strong blur: blur several times to fully obscure
                roi = cv2.GaussianBlur(roi, (k, k), 0)
                roi = cv2.GaussianBlur(roi, (k, k), 0)
                out[y:y2, x:x2] = roi
                cv2.rectangle(out, (x, y), (x2, y2), (0, 200, 255), 2)
                cv2.putText(out, "PRIVACY: face blurred", (x, max(20, y - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
                return out

        # no bbox -> blur the whole frame
        out = cv2.GaussianBlur(out, (k, k), 0)
        cv2.putText(out, "PRIVACY MODE", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
        return out

    def trigger(self, alert_type, frame, record, face_bbox=None):
        """Record a violation. face_bbox is a NEW (optional) parameter telling
        which face region to blur in privacy mode. Old calls without face_bbox
        still work normally."""
        now = time.time()
        last = self.last_trigger.get(alert_type, 0.0)
        if now - last < self.cooldown_seconds:
            return None
        self.last_trigger[alert_type] = now

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_type = alert_type.lower().replace(" ", "_")
        suffix = "_priv" if self.privacy_mode else ""
        image_path = self.image_dir / f"{safe_type}_{timestamp}{suffix}.jpg"

        # create the evidence image
        if self.privacy_mode:
            evidence = self._blur_face(frame, face_bbox)
        else:
            evidence = frame.copy()

        label = f"{alert_type} | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        cv2.putText(evidence, label, (10, evidence.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
        cv2.imwrite(str(image_path), evidence)

        with self.csv_path.open("a", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                record.get("student_id", ""),
                alert_type,
                record.get("state", ""),
                f"{record.get('attention_score', 0):.2f}",
                f"{record.get('liveness_score', 0):.3f}",
                f"{record.get('face_similarity', 0):.3f}",
                f"{record.get('ear', 0):.4f}",
                f"{record.get('mar', 0):.4f}",
                f"{record.get('yaw', 0):.2f}",
                f"{record.get('pitch', 0):.2f}",
                str(image_path),
            ])
        return image_path