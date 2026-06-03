import time

import cv2

from src.alert_system_v2 import AlertSystem
from src.behavior_analyzer.attention_score import AttentionScorer
from src.behavior_analyzer.decision_engine import DecisionEngine
from src.behavior_analyzer.pattern_detector import PatternDetector
from src.behavior_analyzer.temporal_buffer import TemporalBuffer
from src.config import CONFIG
from src.dashboard import Dashboard
from src.database import StudentDatabase
from src.eye_monitor import EyeMonitor
from src.face_detector import FaceDetector
from src.head_pose import HeadPoseEstimator
from src.identity_verifier import IdentityVerifier
from src.liveness_detector import LivenessDetector
from src.mouth_monitor import MouthMonitor
from src.state_classifier import StateClassifier


def build_default_record(student_id, state="ABSENT"):
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
    fps_time = last_time
    fps_frames = 0
    fps = 0.0
    last_pattern_eval = 0.0
    active_patterns = []

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

        detection = face_detector.detect(frame)
        record = build_default_record(student_id)
        record["timestamp"] = now
        record["fps"] = fps

        if detection.face_count == 0:
            state = "ABSENT"
            liveness_status = "NO_FACE"
            identity_status = identity_load_status if identity_load_status != "READY" else "NO_FACE"
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
                color = (0, 220, 80) if state == "OK" else (0, 0, 255)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            if detection.face_count > 1:
                active_patterns.append("Multiple Faces Detected")

        raw_score, display_score = attention.update(state, dt)
        record["state"] = state
        record["attention_score"] = display_score
        buffer.add(record.copy())

        if now - last_pattern_eval >= CONFIG["pattern_interval"]:
            active_patterns = pattern_detector.detect(buffer)
            last_pattern_eval = now

        level, alerts = decision_engine.decide(record, active_patterns)
        if detection.face_count > 1 and "Multiple Faces Detected" not in alerts:
            alerts.append("Multiple Faces Detected")

        for alert in alerts:
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
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
