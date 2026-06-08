import argparse
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
from src.states import ABSENT, BODY_ONLY, OK, PHONE_USAGE


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
    parser = argparse.ArgumentParser(description="E-Proctoring System")
    parser.add_argument("--video", default=None, help="Path to video file (omit for webcam mode)")
    parser.add_argument("--labels", default=None, help="Path to labels_segment CSV for video evaluation")
    parser.add_argument("--student", default=None, help="Student ID (skips the prompt)")
    parser.add_argument("--show", action="store_true", help="Show video frames while processing (video mode only)")
    args = parser.parse_args()
    video_mode = args.video is not None

    CONFIG["evidence_dir"].mkdir(parents=True, exist_ok=True)
    CONFIG["evidence_images_dir"].mkdir(parents=True, exist_ok=True)
    CONFIG["database_csv"].parent.mkdir(parents=True, exist_ok=True)

    if args.student:
        student_id = args.student.strip()
    else:
        student_id = input("Enter student ID: ").strip()
    database = StudentDatabase(CONFIG["database_csv"])
    student = database.get(student_id)

    face_detector = FaceDetector()
    head_pose = HeadPoseEstimator()
    eye_monitor = EyeMonitor(
        CONFIG["ear_threshold"],
        CONFIG["sleep_duration"],
        CONFIG.get("sleep_use_min_eye_ear", True),
        CONFIG.get("sleep_min_eye_threshold_factor", 0.9),
    )
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
        CONFIG.get("liveness_spoof_confirm_seconds", 4.0),
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

    if video_mode:
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            print(f"[ERROR] Cannot open video: {args.video}")
            return 1
        fps_video = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        video_stem = Path(args.video).stem
        output_video_dir = PROJECT_ROOT / "output" / "video_analysis" / video_stem
        output_video_dir.mkdir(parents=True, exist_ok=True)
        output_video_path = output_video_dir / f"{video_stem}_annotated.mp4"
        video_writer = cv2.VideoWriter(
            str(output_video_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps_video,
            (frame_w, frame_h),
        )
        show_window = args.show
        window_title = f"E-Proctoring — {Path(args.video).name}"
        print(f"[INFO] Video mode: {args.video}  ({fps_video:.1f} fps)")
        print(f"[INFO] Output video: {output_video_path}")
    else:
        cap = cv2.VideoCapture(CONFIG["camera_id"])
        if not cap.isOpened():
            print("[ERROR] Webcam not found or cannot be opened.")
            return 1
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CONFIG["frame_width"])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG["frame_height"])
        fps_video = None
        video_writer = None
        show_window = True
        window_title = "Intelligent E-Proctoring System"

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
    person_detections = []
    phone_last_seen: float | None = None   # timestamp of last YOLO phone detection
    last_multi_face_alert = 0.0

    # --- Per-student EAR calibration ---
    # Collect open-eye EAR readings for the first N seconds while the student
    # looks straight at the camera, then set a personal threshold proportional
    # to their baseline.  This handles students with naturally small eyes.
    _calib_seconds = CONFIG.get("ear_calibration_seconds", 4.0)
    _calib_factor  = CONFIG.get("ear_calibration_factor",  0.72)
    _calib_min_open_ear = CONFIG.get("ear_calibration_min_open_ear", 0.14)
    _calib_min_threshold = CONFIG.get("ear_calibration_min_threshold", 0.16)
    _calib_ears: list[float] = []
    _calibrated = False

    print("[INFO] System started. Press q or ESC to exit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            if not video_mode:
                print("[ERROR] Failed to read frame from webcam.")
            break

        if video_mode:
            now = frame_idx / fps_video          # frame-aligned timestamp
            dt = 1.0 / fps_video
            fps = fps_video
        else:
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
            phone_detections, person_detections = phone_detector.detect_people_and_phones(frame)

        detection = face_detector.detect(frame)
        record = build_default_record(student_id)
        record["timestamp"] = now
        record["fps"] = fps

        # --- EAR calibration phase ---
        elapsed = now - session_start if not video_mode else now
        if not _calibrated:
            if detection.face_count > 0:
                _lms = detection.primary_landmarks()
                _y, _p, _ = head_pose.estimate(frame, _lms)
                _e, _ = eye_monitor.check(_lms, now)
                # Only collect while roughly frontal (avoids skewed blink frames)
                if abs(_y) < 20 and abs(_p) < 20 and _e >= _calib_min_open_ear:
                    _calib_ears.append(_e)
            if elapsed >= _calib_seconds:
                if len(_calib_ears) >= 15:
                    import numpy as np
                    baseline = float(np.percentile(_calib_ears, 85))
                    new_thresh = round(max(_calib_min_threshold, baseline * _calib_factor), 3)
                    eye_monitor.ear_threshold = new_thresh
                    liveness_detector.ear_threshold = new_thresh  # sync blink gate
                    print(f"[CALIB] EAR baseline={baseline:.3f}  "
                          f"threshold set to {new_thresh:.3f} "
                          f"(factor {_calib_factor})")
                else:
                    print("[CALIB] Not enough frontal frames — keeping default threshold.")
                _calibrated = True
            else:
                # Draw calibration overlay
                remaining = int(_calib_seconds - elapsed) + 1
                cv2.rectangle(frame, (0, 0), (frame.shape[1], 50), (20, 20, 20), -1)
                cv2.putText(frame,
                            f"Calibrating... Look straight at camera  ({remaining}s)",
                            (20, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 255), 2)

        if detection.face_count == 0:
            # Distinguish body-visible (BODY_ONLY) from fully absent (ABSENT).
            # If YOLO sees a person but mediapipe lost the face, the student is
            # present but their face is off-camera or occluded.
            state = BODY_ONLY if person_detections else ABSENT
            liveness_status = "NO_FACE"
            identity_status = identity_load_status if identity_load_status != "READY" else "NO_FACE"
            # Don't reset phone_last_seen here — a phone stays visible even
            # when the face briefly leaves the frame.
        else:
            landmarks = detection.primary_landmarks()
            yaw, pitch, roll = head_pose.estimate(frame, landmarks)
            ear, sleeping = eye_monitor.check(landmarks, now)
            mar, talking = mouth_monitor.check(landmarks, now)
            liveness_score, liveness_status = liveness_detector.update(now, landmarks, ear, yaw, pitch)
            # Closed eyes / drowsiness are expected to be still. Do not let
            # the anti-spoofing heuristic compete with sleeping detection.
            closed_eye_gate = max(
                eye_monitor.ear_threshold,
                CONFIG.get("liveness_closed_eye_ear_threshold", 0.16),
            )
            if (
                ear < closed_eye_gate
                or getattr(eye_monitor, "last_sleep_ear", ear) < closed_eye_gate
                or sleeping
            ) and liveness_status == "SPOOFING":
                liveness_status = "LIVE"

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

            # Phone detection — "last seen" window: trigger PHONE_USAGE if a
            # phone was detected by YOLO at any point in the last N seconds.
            # Much more robust than a continuous-timer that resets on every
            # missed frame when YOLO detection flickers.
            _phone_window = CONFIG.get("phone_seen_window", 0.5)
            if phone_detections:
                phone_last_seen = now
            if phone_last_seen is not None and (now - phone_last_seen) <= _phone_window:
                state = PHONE_USAGE

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
            "timestamp": round(now if video_mode else (now - session_start), 3),
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
        if video_writer is not None:
            video_writer.write(frame)
        if show_window:
            cv2.imshow(window_title, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if cv2.getWindowProperty(window_title, cv2.WND_PROP_VISIBLE) < 1:
                break

    cap.release()
    if video_writer is not None:
        video_writer.release()
        print(f"[INFO] Annotated video saved: {output_video_path}")
    cv2.destroyAllWindows()
    print("[INFO] System stopped.")

    # --- write session details CSV and run analytics ---
    if detail_rows:
        if video_mode:
            session_dir = PROJECT_ROOT / "output" / "video_analysis" / video_stem
            detail_csv = session_dir / f"{video_stem}_details.csv"
        else:
            session_dir = PROJECT_ROOT / "output" / "webcam_session"
            ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            detail_csv = session_dir / f"session_{ts_str}_details.csv"

        session_dir.mkdir(parents=True, exist_ok=True)
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
            run_post_analytics(detail_csv, args.labels if video_mode else None)
        except Exception as exc:
            print(f"[WARN] Post-analysis failed: {exc}")

    return 0


if __name__ == "__main__":
    raise SystemExit(run())
