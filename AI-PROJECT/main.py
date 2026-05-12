"""
E-Proctoring System — Attention Monitoring
Phát hiện sinh viên làm việc riêng trong giờ học
"""

import cv2
import time
import os
from datetime import datetime
from face_detector import FaceDetector
from head_pose import HeadPoseEstimator
from eye_monitor import EyeMonitor
from mouth_monitor import MouthMonitor
from alert_system import AlertSystem


# ===================== CẤU HÌNH =====================
CONFIG = {
    "yaw_threshold": 18,        # Head yaw change from normal pose -> looking away
    "pitch_threshold": 12,      # Head pitch change from normal pose -> looking down/up
    "ear_threshold": 0.22,      # EAR thấp hơn mức này → đang nhắm mắt
    "sleep_duration": 3.0,      # Giây nhắm mắt liên tục → ngủ gật
    "mar_threshold": 0.50,      # MAR cao hơn mức này → đang nói chuyện
    "talk_duration": 2.0,       # Giây miệng mở liên tục → nói chuyện
    "absent_duration": 3.0,     # Giây không thấy mặt → bỏ ra ngoài
    "multiple_face_duration": 1.5, # Giây thấy hơn 1 mặt liên tục → nhiều người
    "distract_duration": 2.0,   # Giây nhìn sang ngang liên tục → mất tập trung
    "talk_yaw_limit": 18,       # Do not check talking when the head is turned too far
    "alert_display_duration": 3.0, # Giây giữ cảnh báo trên màn hình để tránh nhấp nháy
    "calibration_frames": 30,   # First stable frames used as normal head pose
    "evidence_dir": "evidence", # Thư mục lưu ảnh bằng chứng
    "camera_id": 0,
}
# ====================================================


def draw_dashboard(frame, results: dict, alerts: list):
    """Vẽ thông tin lên màn hình"""
    h, w = frame.shape[:2]

    # Panel nền
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (280, h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    # Tiêu đề
    cv2.putText(frame, "E-PROCTORING", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
    cv2.line(frame, (10, 38), (270, 38), (0, 200, 255), 1)

    # Các chỉ số
    y = 65
    metrics = [
        ("Faces",    str(results.get("face_count", 0))),
        ("Yaw",      f"{results.get('yaw', 0):+.1f} deg"),
        ("YawD",     f"{results.get('yaw_delta', 0):+.1f} deg"),
        ("Pitch",    f"{results.get('pitch', 0):+.1f} deg"),
        ("PitchD",   f"{results.get('pitch_delta', 0):+.1f} deg"),
        ("EAR",      f"{results.get('ear', 0):.3f}"),
        ("MAR",      f"{results.get('mar', 0):.3f}"),
        ("Status",   results.get("status", "OK")),
    ]
    for label, value in metrics:
        color = (0, 255, 100) if results.get("status") == "OK" else (0, 100, 255)
        if label == "Status":
            color = (0, 255, 100) if value == "OK" else (0, 80, 255)
        cv2.putText(frame, f"{label}: {value}", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1)
        y += 26

    # Cảnh báo
    if alerts:
        y += 10
        cv2.putText(frame, "! ALERT:", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 60, 255), 2)
        y += 24
        for alert in alerts[-3:]:
            cv2.putText(frame, f"  {alert}", (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 80, 255), 1)
            y += 20

    # Timestamp
    ts = datetime.now().strftime("%H:%M:%S")
    cv2.putText(frame, ts, (w - 90, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)


def main():
    os.makedirs(CONFIG["evidence_dir"], exist_ok=True)

    cap = cv2.VideoCapture(CONFIG["camera_id"])
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    detector   = FaceDetector()
    head_pose  = HeadPoseEstimator()
    eye_mon    = EyeMonitor(CONFIG)
    mouth_mon  = MouthMonitor(CONFIG)
    alert_sys  = AlertSystem(CONFIG)

    last_seen  = time.time()
    fps_time   = time.time()
    frame_count = 0
    multiple_started = None
    yaw_started = None
    down_started = None
    yaw_baseline = None
    yaw_samples = []
    pitch_baseline = None
    pitch_samples = []
    recent_alerts = {}

    def show_alert(key: str, message: str):
        recent_alerts[key] = (
            message,
            time.time() + CONFIG["alert_display_duration"],
        )

    print("[INFO] System started. Press 'q' or ESC to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        now = time.time()
        frame_count += 1
        results = {"status": "OK"}

        # 1. Phát hiện khuôn mặt
        faces, landmarks_list = detector.detect(frame)
        reliable_face_count = len(landmarks_list)
        results["face_count"] = reliable_face_count

        # Cảnh báo nhiều người
        if reliable_face_count > 1:
            if multiple_started is None:
                multiple_started = now
            elapsed = now - multiple_started
            if elapsed >= CONFIG["multiple_face_duration"]:
                show_alert("multiple_faces", "Multiple faces detected!")
                alert_sys.trigger("multiple_faces", frame, CONFIG["evidence_dir"])
        else:
            multiple_started = None
            recent_alerts.pop("multiple_faces", None)

        # Không thấy mặt
        if reliable_face_count == 0:
            yaw_started = None
            down_started = None
            elapsed = now - last_seen
            if elapsed >= CONFIG["absent_duration"]:
                show_alert("absent", f"No face detected ({elapsed:.1f}s)")
                alert_sys.trigger("absent", frame, CONFIG["evidence_dir"])
            results["status"] = "ABSENT"
        else:
            last_seen = now

            # Lấy mặt đầu tiên để phân tích
            landmarks = landmarks_list[0] if landmarks_list else None

            if landmarks is not None:
                # 2. Eye Monitor — compute first so closed eyes are not treated as looking down.
                ear, sleeping = eye_mon.check(landmarks)
                eyes_open = ear >= CONFIG["ear_threshold"]
                results["ear"] = ear

                # 3. Head Pose — phát hiện nhìn sang ngang / cúi đầu
                yaw, pitch, roll = head_pose.estimate(frame, landmarks)
                results["yaw"]   = yaw
                results["pitch"] = pitch
                results["yaw_delta"] = 0.0
                results["pitch_delta"] = 0.0

                if eyes_open:
                    if yaw_baseline is None:
                        yaw_samples.append(yaw)
                        if len(yaw_samples) >= CONFIG["calibration_frames"]:
                            yaw_baseline = sum(yaw_samples) / len(yaw_samples)
                    if pitch_baseline is None:
                        pitch_samples.append(pitch)
                        if len(pitch_samples) >= CONFIG["calibration_frames"]:
                            pitch_baseline = sum(pitch_samples) / len(pitch_samples)

                if yaw_baseline is not None:
                    results["yaw_delta"] = yaw - yaw_baseline
                if pitch_baseline is not None:
                    results["pitch_delta"] = pitch - pitch_baseline

                if yaw_baseline is not None and abs(results["yaw_delta"]) > CONFIG["yaw_threshold"]:
                    if yaw_started is None:
                        yaw_started = now
                    elapsed = now - yaw_started
                    if elapsed >= CONFIG["distract_duration"]:
                        show_alert("distracted", f"Looking away ({results['yaw_delta']:+.0f}°)")
                        alert_sys.trigger("distracted", frame, CONFIG["evidence_dir"])
                        results["status"] = "DISTRACTED"
                else:
                    yaw_started = None

                if (
                    pitch_baseline is not None
                    and eyes_open
                    and abs(results["pitch_delta"]) > CONFIG["pitch_threshold"]
                ):
                    if down_started is None:
                        down_started = now
                    elapsed = now - down_started
                    if elapsed >= CONFIG["distract_duration"]:
                        show_alert("looking_down", f"Looking down ({results['pitch_delta']:+.0f}°) - Phone?")
                        alert_sys.trigger("looking_down", frame, CONFIG["evidence_dir"])
                        results["status"] = "DISTRACTED"
                else:
                    down_started = None

                if sleeping:
                    recent_alerts.pop("looking_down", None)
                    show_alert("sleeping", "Drowsiness detected!")
                    alert_sys.trigger("sleeping", frame, CONFIG["evidence_dir"])
                    results["status"] = "SLEEPING"

                # 4. Mouth Monitor — only check talking when the face is mostly frontal.
                yaw_for_talking = abs(results.get("yaw_delta", yaw))
                if yaw_baseline is not None and yaw_for_talking <= CONFIG["talk_yaw_limit"]:
                    mar, talking = mouth_mon.check(landmarks)
                    results["mar"] = mar
                    if talking:
                        show_alert("talking", "Talking detected!")
                        alert_sys.trigger("talking", frame, CONFIG["evidence_dir"])
                        results["status"] = "TALKING"
                else:
                    mouth_mon._open_since = None
                    recent_alerts.pop("talking", None)
                    results["mar"] = 0

        # Vẽ bounding boxes
        for (x, y, w, h) in faces:
            color = (0, 255, 0) if results["status"] == "OK" else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

        # FPS
        if frame_count % 30 == 0:
            fps = 30 / (time.time() - fps_time)
            fps_time = time.time()
            results["fps"] = fps

        recent_alerts = {
            key: value for key, value in recent_alerts.items()
            if value[1] >= now
        }
        alerts = [message for message, _ in recent_alerts.values()]

        draw_dashboard(frame, results, alerts)
        cv2.imshow("E-Proctoring System", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        if cv2.getWindowProperty("E-Proctoring System", cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] System stopped.")


if __name__ == "__main__":
    main()
