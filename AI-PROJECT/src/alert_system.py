import csv
import time
from datetime import datetime
from pathlib import Path

import cv2


class AlertSystem:
    def __init__(self, csv_path: Path, image_dir: Path, cooldown_seconds=5.0):
        self.csv_path = Path(csv_path)
        self.image_dir = Path(image_dir)
        self.cooldown_seconds = cooldown_seconds
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

    def trigger(self, alert_type, frame, record):
        now = time.time()
        last = self.last_trigger.get(alert_type, 0.0)
        if now - last < self.cooldown_seconds:
            return None
        self.last_trigger[alert_type] = now

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_type = alert_type.lower().replace(" ", "_")
        image_path = self.image_dir / f"{safe_type}_{timestamp}.jpg"

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
