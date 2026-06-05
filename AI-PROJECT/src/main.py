import csv
import time
from datetime import datetime
from pathlib import Path

import cv2

from src.alert_system_v2 import AlertSystem
from src.behavior_analyzer.attention_score import AttentionScorer
from src.behavior_analyzer.decision_engine import DecisionEngine
from src.behavior_analyzer.pattern_detector import PatternDetector
from src.behavior_analyzer.temporal_buffer import TemporalBuffer
from src.config import CONFIG, PROJECT_ROOT
from src.dashboard import Dashboard
from src.database import StudentDatabase
from src.eye_monitor import EyeMonitor
from src.face_detector import FaceDetector
from src.head_pose import HeadPoseEstimator
from src.identity_verifier import IdentityVerifier
from src.liveness_detector import LivenessDetector
from src.mouth_monitor import MouthMonitor
from src.phone_detector import PhoneDetector, expanded_intersection
from src.state_classifier import StateClassifier
from src.states import ABSENT, OK, PHONE_USAGE


def build_default_record(student_id, state=ABSENT):
    return {
        "timestamp": time.time(),
        "student_id": student_id,
        "state": state,
        "ear": 0.0,
        "mar": 0.0,
        "yaw": 0.0,
        "pitch": 0.0,
        "roll": 0.0,
        "liveness_score": 0.0,
        "liveness_status": "NO_FACE",
        "identity_status": "NOT_CHECKED",
        "face_similarity": 0.0,
        "attention_score": 100.0,
        "fps": 0.0,
    }


def run():
    CONFIG["evidence_dir"].mkdir(parents=True, exist_ok=True)
    CONFIG["evidence_images_dir"].mkdir(parents=True, exist_ok=True)
    CONFIG["database_csv"].parent.mkdir(parents=True, exist_ok=True)

    student_id = input("Enter student ID: ").strip()
    database = StudentDatabase(CONFIG["database_csv"])
    student = database.get(student_id)

    face_detector = FaceDetector()
    head_pose = HeadPoseEstimator()
    eye_monitor = EyeMonitor(CONFIG["ear_threshold"], CONFIG["sleep_duration"])
    mouth_monitor = MouthMonitor(
        CONFIG["mar_threshold"], CONFIG["talk_duration"],
        talk_window=CONFIG.get("talk_window", 1.5),
        talk_min_transitions=CONFIG.get("talk_min_transitions", 3),
        talk_mar_variance_threshold=CONFIG.get("talk_mar_variance_threshold", 0.005),
    )
    state_classifier = StateClassifier(CONFIG["yaw_threshold"], CONFIG["pitch_down_threshold"])
    liveness_detector = LivenessDetector(
        CONFIG["liveness_threshold"],
        CONFIG["liveness_warmup_seconds"],
        CONFIG["liveness_window_seconds"],
        CONFIG["ear_threshold"],
    )
    identity_verifier = IdentityVerifier(face_detector, CONFIG["identity_similarity_threshold"])
    attention = AttentionScorer(CONFIG["attention_rates"], CONFIG["attention_alpha"])
    buffer = TemporalBuffer(CONFIG["buffer_seconds"])
    pattern_detector = PatternDetector()
    decision_engine = DecisionEngine()
    alert_system = AlertSystem(CONFIG["violations_csv"], CONFIG["evidence_images_dir"], privacy_mode=True)
    dashboard = Dashboard()
    phone_detector = PhoneDetector(
        CONFIG.get("phone_model_path", "yolov8n.pt"),
        CONFIG.get("phone_confidence", 0.35),
        CONFIG.get("phone_detection_enabled", True),
    )
    if phone_detector.warning:
        print(f"[WARN] {phone_detector.warning}")

    identity_load_status = "NOT_LOADED"
    identity_load_message = ""
    if not student_id:
        identity_load_status = "UNKNOWN_ID"
        identity_load_message = "Empty student ID"
    elif student is None:
        identity_load_status = "UNKNOWN_ID"
        identity_load_message = f"Unknown student ID: {student_id}"
    else:
        ok, message = identity_verifier.load_reference(student.reference_image)
        identity_load_status = "READY" if ok else "NO_REFERENCE"
        identity_load_message = message

    print(f"[INFO] Student ID: {student_id or '(empty)'}")
    print(f"[INFO] Identity database status: {identity_load_status} - {identity_load_message}")

    cap = cv2.VideoCapture(CONFIG["camera_id"])
    if not cap.isOpened():
        print("[ERROR] Webcam not found or cannot be opened.")
        return 1

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CONFIG["frame_width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG["frame_height"])

    last_time = time.time()
    session_start = last_time
    fps_time = last_time
    fps_frames = 0
    fps = 0.0
    last_pattern_eval = 0.0
    active_patterns = []
    detail_rows = []
    frame_idx = 0
    phone_detections = []
    phone_started_at = None
    last_multi_face_alert = 0.0

    print("[INFO] System started. Press q or ESC to exit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            print("[ERROR] Failed to read frame from webcam.")
            break

        now = time.time()
        dt = max(0.001, min(now - last_time, 1.0))
        last_time = now
        fps_frames += 1
        if now - fps_time >= 1.0:
            fps = fps_frames / (now - fps_time)
            fps_time = now
            fps_frames = 0

        frame_idx += 1
        if phone_detector.available and frame_idx % CONFIG.get("phone_interval_frames", 5) == 0:
            phone_detections = phone_detector.detect(frame)

        detection = face_detector.detect(frame)
        record = build_default_record(student_id)
        record["timestamp"] = now
        record["fps"] = fps

        if detection.face_count == 0:
            state = ABSENT
            liveness_status = "NO_FACE"
            identity_status = identity_load_status if identity_load_status != "READY" else "NO_FACE"
            phone_started_at = None
        else:
            landmarks = detection.primary_landmarks()
            yaw, pitch, roll = head_pose.estimate(frame, landmarks)
            ear, sleeping = eye_monitor.check(landmarks, now)
            mar, talking = mouth_monitor.check(landmarks, now)
            liveness_score, liveness_status = liveness_detector.update(now, landmarks, ear, yaw, pitch)

            if identity_load_status != "READY":
                identity_status = identity_load_status
                face_similarity = 0.0
            elif liveness_status == "SPOOFING":
                identity_status = "BLOCKED"
                face_similarity = 0.0
            else:
                identity_result = identity_verifier.verify(landmarks)
                identity_status = identity_result.status
                face_similarity = identity_result.similarity

            state = state_classifier.classify(True, yaw, pitch, sleeping, talking)

            # phone proximity check
            if phone_detections:
                raw_phone = any(
                    expanded_intersection(fb, phone.bbox,
                                          CONFIG.get("phone_near_student_scale", 1.8))
                    for fb in detection.faces
                    for phone in phone_detections
                )
                if raw_phone:
                    if phone_started_at is None:
                        phone_started_at = now
                    if (now - phone_started_at) >= CONFIG.get("video_phone_duration", 0.5):
                        state = PHONE_USAGE
                else:
                    phone_started_at = None

            record.update({
                "ear": ear,
                "mar": mar,
                "yaw": yaw,
                "pitch": pitch,
                "roll": roll,
                "liveness_score": liveness_score,
                "liveness_status": liveness_status,
                "identity_status": identity_status,
                "face_similarity": face_similarity,
            })

            for x, y, w, h in detection.faces:
                color = (0, 220, 80) if state == OK else (0, 0, 255)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        for phone in phone_detections:
            px, py, pw, ph = phone.bbox
            cv2.rectangle(frame, (px, py), (px + pw, py + ph), (255, 0, 255), 2)
            cv2.putText(frame, f"PHONE {phone.confidence:.2f}", (px, max(20, py - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 255), 2)

        raw_score, display_score = attention.update(state, dt)
        record["state"] = state
        record["attention_score"] = display_score
        buffer.add(record.copy())
        detail_rows.append({
            "timestamp": round(now - session_start, 3),
            "student": student_id or "Unknown",
            "state": record["state"],
            "ear": record["ear"],
            "mar": record["mar"],
            "yaw": record["yaw"],
            "pitch": record["pitch"],
            "roll": record.get("roll", 0.0),
            "attention_score": record["attention_score"],
            "phone_detected": state == PHONE_USAGE,
            "patterns": "; ".join(active_patterns),
        })

        if now - last_pattern_eval >= CONFIG["pattern_interval"]:
            active_patterns = pattern_detector.detect(buffer)
            last_pattern_eval = now

        level, alerts = decision_engine.decide(record, active_patterns)
        if detection.face_count > 1 and "Multiple Faces Detected" not in alerts:
            alerts.append("Multiple Faces Detected")

        for alert in alerts:
            if alert == "Multiple Faces Detected":
                if now - last_multi_face_alert >= 5.0:
                    alert_system.trigger(alert, frame, record)
                    last_multi_face_alert = now
            else:
                alert_system.trigger(alert, frame, record)

        dashboard.draw(frame, record, alerts)
        cv2.imshow("Intelligent E-Proctoring System", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if cv2.getWindowProperty("Intelligent E-Proctoring System", cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] System stopped.")

    # --- write session details CSV and run analytics ---
    if detail_rows:
        session_dir = PROJECT_ROOT / "output" / "webcam_session"
        session_dir.mkdir(parents=True, exist_ok=True)
        ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        detail_csv = session_dir / f"session_{ts_str}_details.csv"

        fieldnames = [
            "timestamp", "student", "state", "ear", "mar", "yaw", "pitch",
            "roll", "attention_score", "phone_detected", "patterns",
        ]
        with open(detail_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(detail_rows)
        print(f"[INFO] Session details saved: {detail_csv}")

        try:
            from src.analytics import run_post_analytics
            run_post_analytics(detail_csv)
        except Exception as exc:
            print(f"[WARN] Post-analysis failed: {exc}")

    return 0


if __name__ == "__main__":
    raise SystemExit(run())
