"""
alert_system.py
Quản lý cảnh báo: lưu ảnh bằng chứng + ghi log vi phạm
"""

import cv2
import csv
import time
import os
from datetime import datetime


# Thời gian tối thiểu giữa 2 lần lưu bằng chứng cùng loại (giây)
COOLDOWN = 5.0


class AlertSystem:
    def __init__(self, config: dict):
        self.evidence_dir = config["evidence_dir"]
        self.log_path = os.path.join(self.evidence_dir, "violations.csv")
        self._last_trigger: dict[str, float] = {}

        os.makedirs(self.evidence_dir, exist_ok=True)

        # Tạo file CSV nếu chưa có
        if not os.path.exists(self.log_path):
            with open(self.log_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "violation_type", "image_path"])

    def trigger(self, violation_type: str, frame, evidence_dir: str = None):
        """
        Kích hoạt cảnh báo: lưu ảnh + ghi log.
        Có cooldown để tránh spam file.
        """
        now = time.time()
        last = self._last_trigger.get(violation_type, 0)

        if now - last < COOLDOWN:
            return  # Cooldown chưa hết

        self._last_trigger[violation_type] = now

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{violation_type}_{ts}.jpg"
        img_path = os.path.join(self.evidence_dir, filename)

        # Thêm watermark timestamp lên ảnh bằng chứng
        evidence = frame.copy()
        label = f"[{violation_type.upper()}] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        cv2.putText(evidence, label, (10, evidence.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

        cv2.imwrite(img_path, evidence)

        # Ghi log CSV
        with open(self.log_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                violation_type,
                img_path,
            ])

        print(f"[ALERT] {violation_type} → {img_path}")

    def get_summary(self) -> dict:
        """Thống kê số lần vi phạm theo từng loại"""
        summary = {}
        if not os.path.exists(self.log_path):
            return summary
        with open(self.log_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                vtype = row["violation_type"]
                summary[vtype] = summary.get(vtype, 0) + 1
        return summary
