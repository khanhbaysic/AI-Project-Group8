from collections import deque
from datetime import datetime

import cv2
import numpy as np

from src.states import (
    ALL_STATES,
    LABEL_VI,
    PHONE_USAGE,
    SLEEPING,
    STATE_COLORS,
    TALKING,
)


def hex_to_bgr(hex_color):
    value = hex_color.lstrip("#")
    r = int(value[0:2], 16)
    g = int(value[2:4], 16)
    b = int(value[4:6], 16)
    return b, g, r


STATE_COLORS_BGR = {state: hex_to_bgr(color) for state, color in STATE_COLORS.items()}


# Large on-screen alert messages for critical behavioral states.
# Each entry: state -> (label, background_color, text_color)
_STATE_ALERTS = {
    SLEEPING:    ("SLEEPING DETECTED",    STATE_COLORS_BGR[SLEEPING], (255, 255, 255)),
    TALKING:     ("TALKING DETECTED",     STATE_COLORS_BGR[TALKING], (255, 255, 255)),
    PHONE_USAGE: ("PHONE USAGE DETECTED", STATE_COLORS_BGR[PHONE_USAGE], (255, 255, 255)),
}


class Dashboard:
    def __init__(self, history_size=120):
        self.score_history = deque(maxlen=history_size)

    def draw(self, frame, record, alerts):
        h, w = frame.shape[:2]
        panel_w = 340
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (panel_w, h), (18, 18, 18), -1)
        cv2.addWeighted(overlay, 0.62, frame, 0.38, 0, frame)

        cv2.putText(frame, "E-PROCTORING", (12, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 220, 255), 2)
        cv2.line(frame, (12, 42), (panel_w - 12, 42), (0, 220, 255), 1)

        score = float(record.get("attention_score", 100))
        self.score_history.append(score)
        score_color = (0, 220, 80) if score >= 70 else (0, 210, 255) if score >= 40 else (0, 80, 255)
        cv2.putText(frame, "Attention Score", (12, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (230, 230, 230), 1)
        cv2.putText(frame, f"{score:05.1f}", (225, 72),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.72, score_color, 2)
        cv2.rectangle(frame, (12, 84), (panel_w - 14, 102), (70, 70, 70), 1)
        cv2.rectangle(frame, (14, 86), (14 + int((panel_w - 30) * score / 100), 100), score_color, -1)

        chart_x, chart_y, chart_w, chart_h = 12, 116, panel_w - 26, 52
        cv2.rectangle(frame, (chart_x, chart_y), (chart_x + chart_w, chart_y + chart_h), (70, 70, 70), 1)
        values = list(self.score_history)
        if len(values) > 1:
            pts = []
            for i, value in enumerate(values):
                x = chart_x + int(i * chart_w / max(1, len(values) - 1))
                y = chart_y + chart_h - int(value * chart_h / 100)
                pts.append((x, y))
            for p1, p2 in zip(pts, pts[1:]):
                cv2.line(frame, p1, p2, score_color, 1)

        state = record.get("state", "")
        state_color = STATE_COLORS_BGR.get(state, (220, 220, 220))
        cv2.rectangle(frame, (12, 180), (panel_w - 14, 222), state_color, -1)
        cv2.putText(frame, "CURRENT STATE", (22, 196),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
        cv2.putText(frame, state or "UNKNOWN", (22, 216),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2)

        legend_y = 242
        cv2.putText(frame, "Legend", (12, legend_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (230, 230, 230), 1)
        for idx, state_name in enumerate(ALL_STATES):
            col = idx % 2
            row = idx // 2
            x = 12 + col * 162
            y0 = legend_y + 14 + row * 22
            color = STATE_COLORS_BGR.get(state_name, (180, 180, 180))
            cv2.rectangle(frame, (x, y0), (x + 12, y0 + 12), color, -1)
            cv2.putText(frame, LABEL_VI.get(state_name, state_name), (x + 18, y0 + 11),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (225, 225, 225), 1)

        metrics = [
            ("Student", record.get("student_id", "")),
            ("State", record.get("state", "")),
            ("Identity", record.get("identity_status", "")),
            ("Similarity", f"{record.get('face_similarity', 0):.3f}"),
            ("Liveness", f"{record.get('liveness_status', '')} {record.get('liveness_score', 0):.2f}"),
            ("Yaw", f"{record.get('yaw', 0):+.1f}"),
            ("Pitch", f"{record.get('pitch', 0):+.1f}"),
            ("EAR", f"{record.get('ear', 0):.3f}"),
            ("MAR", f"{record.get('mar', 0):.3f}"),
            ("FPS", f"{record.get('fps', 0):.1f}"),
        ]
        y = 342
        for label, value in metrics:
            cv2.putText(frame, f"{label}: {value}", (12, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, (230, 230, 230), 1)
            y += 25

        if alerts:
            y += 10
            cv2.putText(frame, "! ALERTS", (12, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 70, 255), 2)
            y += 25
            for alert in alerts[-5:]:
                cv2.putText(frame, f"- {alert}", (12, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 100, 255), 1)
                y += 20

        # --- prominent center-screen alert for SLEEPING / TALKING / PHONE ---
        if state in _STATE_ALERTS:
            self._draw_alert_banner(frame, w, h, state)

        cv2.putText(frame, datetime.now().strftime("%H:%M:%S"), (w - 95, h - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (190, 190, 190), 1)
        return frame

    # ------------------------------------------------------------------

    @staticmethod
    def _draw_alert_banner(frame, w, h, state):
        """Draw a large translucent alert banner across the top-center."""
        label, bg_color, text_color = _STATE_ALERTS[state]

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        thickness = 3
        (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)

        # banner geometry
        pad_x, pad_y = 32, 18
        bw = tw + pad_x * 2
        bh = th + baseline + pad_y * 2
        bx = (w - bw) // 2
        by = 12

        # translucent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (bx, by), (bx + bw, by + bh), bg_color, -1)
        cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)

        # border
        cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), text_color, 2)

        # warning icon (triangle outline)
        icon_x = bx + pad_x - 4
        icon_cy = by + pad_y + th // 2
        tri_size = 14
        pts_tri = [
            (icon_x, icon_cy + tri_size),
            (icon_x + tri_size, icon_cy - tri_size),
            (icon_x + 2 * tri_size, icon_cy + tri_size),
        ]
        cv2.polylines(frame, [np.array(pts_tri)], True, text_color, 2, cv2.LINE_AA)
        cv2.putText(frame, "!", (icon_x + tri_size - 4, icon_cy + tri_size - 5),
                    font, 0.55, text_color, 2, cv2.LINE_AA)

        # text
        tx = icon_x + 2 * tri_size + 12
        ty = by + pad_y + th
        cv2.putText(frame, label, (tx, ty), font, font_scale, text_color,
                    thickness, cv2.LINE_AA)
