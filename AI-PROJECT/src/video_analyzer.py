import argparse
import csv
import time
from pathlib import Path

import cv2

from src.config import CONFIG, PROJECT_ROOT
from src.face_detector import FaceDetector
from src.head_pose import HeadPoseEstimator
from src.phone_detector import PhoneDetector, expanded_intersection
from src.state_classifier import StateClassifier
from src.student_state import StudentState
from src.video_tracker import CentroidTracker


OUTPUT_DIR = PROJECT_ROOT / "output" / "video_analysis"


def color_for_state(state):
    return {
        "OK": (0, 220, 80),
        "DISTRACTED": (0, 200, 255),
        "TALKING": (255, 180, 0),
        "SLEEPING": (0, 80, 255),
        "ABSENT": (80, 80, 80),
        "PHONE_USAGE": (255, 0, 255),
        "BODY_ONLY": (180, 180, 180),
    }.get(state, (220, 220, 220))


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def bbox_area(bbox):
    return max(0, bbox[2]) * max(0, bbox[3])


def bbox_center(bbox):
    x, y, w, h = bbox
    return x + w / 2.0, y + h / 2.0


def center_inside(inner_bbox, outer_bbox):
    cx, cy = bbox_center(inner_bbox)
    return point_inside_bbox((cx, cy), outer_bbox)


def point_inside_bbox(point, bbox):
    px, py = point
    x, y, w, h = bbox
    return x <= px <= x + w and y <= py <= y + h


def filter_people(people, width, height, min_area_ratio):
    min_area = width * height * min_area_ratio
    return [person for person in people if bbox_area(person.bbox) >= min_area]


def find_face_for_person(person_bbox, face_bboxes, used_face_indices):
    candidates = [
        (bbox_area(face_bbox), face_idx)
        for face_idx, face_bbox in enumerate(face_bboxes)
        if face_idx not in used_face_indices and center_inside(face_bbox, person_bbox)
    ]
    if not candidates:
        return None
    return max(candidates)[1]


def normalized_distance_to_bbox_center(point, bbox):
    cx, cy = bbox_center(bbox)
    _, _, w, h = bbox
    scale = max(w, h, 1)
    return ((point[0] - cx) ** 2 + (point[1] - cy) ** 2) ** 0.5 / scale


def assign_phones_to_people(person_bboxes, phone_detections):
    owners = set()
    for phone in phone_detections:
        phone_center = bbox_center(phone.bbox)
        candidates = [
            (normalized_distance_to_bbox_center(phone_center, bbox), idx)
            for idx, bbox in enumerate(person_bboxes)
            if point_inside_bbox(phone_center, bbox)
        ]
        if not candidates:
            candidates = [
                (normalized_distance_to_bbox_center(phone_center, bbox), idx)
                for idx, bbox in enumerate(person_bboxes)
                if expanded_intersection(bbox, phone.bbox, 1.05)
            ]
        if candidates:
            owners.add(min(candidates)[1])
    return owners


def analyze_video(video_path: Path, show=False, output_dir=OUTPUT_DIR):
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
    dt = 1.0 / max(fps, 1.0)

    output_video = output_dir / f"{video_path.stem}_annotated.mp4"
    detail_csv = output_dir / f"{video_path.stem}_details.csv"
    summary_csv = output_dir / f"{video_path.stem}_summary.csv"

    writer = cv2.VideoWriter(
        str(output_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    detector = FaceDetector(
        max_num_faces=20,
        detection_scale=CONFIG.get("video_detection_scale", 1.0),
    )
    face_tracker = CentroidTracker(max_distance=max(width, height) * 0.08, max_missed=int(fps * 2))
    person_tracker = CentroidTracker(max_distance=max(width, height) * 0.12, max_missed=int(fps * 2))
    head_pose = HeadPoseEstimator()
    pitch_threshold = CONFIG["pitch_down_threshold"] if CONFIG.get("video_use_pitch_distraction", False) else -999.0
    classifier = StateClassifier(CONFIG.get("video_yaw_threshold", CONFIG["yaw_threshold"]), pitch_threshold)
    video_config = dict(CONFIG)
    video_config["ear_threshold"] = CONFIG.get("video_ear_threshold", CONFIG["ear_threshold"])
    video_config["sleep_duration"] = CONFIG.get("video_sleep_duration", CONFIG["sleep_duration"])
    video_config["progressive_drowsiness_ear_threshold"] = CONFIG.get(
        "video_progressive_drowsiness_ear_threshold",
        video_config["ear_threshold"],
    )
    phone_detector = PhoneDetector(
        CONFIG.get("phone_model_path", "yolov8n.pt"),
        CONFIG.get("phone_confidence", 0.35),
        CONFIG.get("phone_detection_enabled", True),
    )
    if phone_detector.warning:
        print(f"[WARN] {phone_detector.warning}")
    phone_detections = []
    person_detections = []
    students: dict[int, StudentState] = {}
    distraction_started_at: dict[int, float] = {}
    phone_started_at: dict[int, float] = {}
    detail_rows = []

    frame_idx = 0
    start = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        timestamp = frame_idx / fps
        detection = detector.detect(frame)
        if phone_detector.available and frame_idx % CONFIG.get("phone_interval_frames", 5) == 0:
            if CONFIG.get("person_tracking_enabled", True):
                phone_detections, person_detections = phone_detector.detect_people_and_phones(frame)
                person_detections = filter_people(
                    person_detections,
                    width,
                    height,
                    CONFIG.get("person_min_box_area_ratio", 0.01),
                )
            else:
                phone_detections = phone_detector.detect(frame)
                person_detections = []

        use_person_tracking = (
            CONFIG.get("person_tracking_enabled", True)
            and phone_detector.available
            and bool(person_detections)
        )

        if use_person_tracking:
            person_bboxes = [person.bbox for person in person_detections]
            phone_owner_indices = assign_phones_to_people(person_bboxes, phone_detections)
            assignments, tracks = person_tracker.update(person_bboxes)
            matched_track_ids = set(assignments.values())
            used_face_indices = set()

            for det_idx, track_id in assignments.items():
                label = f"Student_{track_id}"
                if track_id not in students:
                    students[track_id] = StudentState(label, video_config)

                bbox = person_bboxes[det_idx]
                face_idx = find_face_for_person(bbox, detection.faces, used_face_indices)
                raw_phone_detected = det_idx in phone_owner_indices
                if raw_phone_detected:
                    phone_started_at.setdefault(track_id, timestamp)
                    phone_detected = timestamp - phone_started_at[track_id] >= CONFIG.get("video_phone_duration", 0.0)
                else:
                    phone_started_at.pop(track_id, None)
                    phone_detected = False

                if face_idx is not None:
                    used_face_indices.add(face_idx)
                    landmarks = detection.landmarks_list[face_idx]
                    face_bbox = detection.faces[face_idx]
                    yaw, pitch, roll = head_pose.estimate(frame, landmarks)
                    ear, sleeping = students[track_id].eye_monitor.check(landmarks, timestamp)
                    mar, talking = students[track_id].mouth_monitor.check(landmarks, timestamp)
                    state = classifier.classify(True, yaw, pitch, sleeping, talking)
                    if state == "DISTRACTED":
                        distraction_started_at.setdefault(track_id, timestamp)
                        if timestamp - distraction_started_at[track_id] < CONFIG.get("video_distraction_duration", 0.0):
                            state = "OK"
                    else:
                        distraction_started_at.pop(track_id, None)
                else:
                    face_bbox = None
                    yaw = pitch = roll = ear = mar = 0.0
                    state = "BODY_ONLY"
                    distraction_started_at.pop(track_id, None)

                if phone_detected:
                    state = "PHONE_USAGE"

                record = {
                    "timestamp": timestamp,
                    "student": label,
                    "state": state,
                    "ear": ear,
                    "mar": mar,
                    "yaw": yaw,
                    "pitch": pitch,
                    "roll": roll,
                    "attention_score": students[track_id].attention.display_score,
                    "phone_detected": phone_detected,
                }
                score, patterns = students[track_id].update(record, dt)
                record["attention_score"] = score
                record["patterns"] = "; ".join(patterns)
                detail_rows.append(record.copy())

                x, y, w, h = bbox
                color = color_for_state(state)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                if face_bbox is not None:
                    fx, fy, fw, fh = face_bbox
                    cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), color, 1)
                cv2.putText(frame, f"{label} {state} {score:.1f}", (x, max(20, y - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        else:
            assignments, tracks = face_tracker.update(detection.faces)
            matched_track_ids = set(assignments.values())

            for det_idx, track_id in assignments.items():
                label = f"Student_{track_id}"
                if track_id not in students:
                    students[track_id] = StudentState(label, video_config)

                landmarks = detection.landmarks_list[det_idx]
                bbox = detection.faces[det_idx]
                yaw, pitch, roll = head_pose.estimate(frame, landmarks)
                ear, sleeping = students[track_id].eye_monitor.check(landmarks, timestamp)
                mar, talking = students[track_id].mouth_monitor.check(landmarks, timestamp)
                state = classifier.classify(True, yaw, pitch, sleeping, talking)
                if state == "DISTRACTED":
                    distraction_started_at.setdefault(track_id, timestamp)
                    if timestamp - distraction_started_at[track_id] < CONFIG.get("video_distraction_duration", 0.0):
                        state = "OK"
                else:
                    distraction_started_at.pop(track_id, None)
                raw_phone_detected = any(
                    expanded_intersection(bbox, phone.bbox, CONFIG.get("phone_near_student_scale", 1.8))
                    for phone in phone_detections
                )
                if raw_phone_detected:
                    phone_started_at.setdefault(track_id, timestamp)
                    phone_detected = timestamp - phone_started_at[track_id] >= CONFIG.get("video_phone_duration", 0.0)
                else:
                    phone_started_at.pop(track_id, None)
                    phone_detected = False
                if phone_detected:
                    state = "PHONE_USAGE"

                record = {
                    "timestamp": timestamp,
                    "student": label,
                    "state": state,
                    "ear": ear,
                    "mar": mar,
                    "yaw": yaw,
                    "pitch": pitch,
                    "roll": roll,
                    "attention_score": students[track_id].attention.display_score,
                    "phone_detected": phone_detected,
                }
                score, patterns = students[track_id].update(record, dt)
                record["attention_score"] = score
                record["patterns"] = "; ".join(patterns)
                detail_rows.append(record.copy())

                x, y, w, h = bbox
                color = color_for_state(state)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, f"{label} {state} {score:.1f}", (x, max(20, y - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        for phone in phone_detections:
            x, y, w, h = phone.bbox
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 255), 2)
            cv2.putText(frame, f"PHONE {phone.confidence:.2f}", (x, max(20, y - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 255), 2)

        for track_id, track in list(tracks.items()):
            if track_id in matched_track_ids or track_id not in students:
                continue
            label = students[track_id].student_label
            record = {
                "timestamp": timestamp,
                "student": label,
                "state": "ABSENT",
                "ear": 0.0,
                "mar": 0.0,
                "yaw": 0.0,
                "pitch": 0.0,
                "roll": 0.0,
                "attention_score": students[track_id].attention.display_score,
                "phone_detected": False,
            }
            score, patterns = students[track_id].update(record, dt)
            record["attention_score"] = score
            record["patterns"] = "; ".join(patterns)
            detail_rows.append(record.copy())

        visible_label = "Persons" if use_person_tracking else "Faces"
        cv2.putText(frame, f"Time: {timestamp:.1f}s | {visible_label}: {len(assignments)} | Tracked: {len(students)} | Phones: {len(phone_detections)}",
                    (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 220, 255), 2)
        writer.write(frame)
        if show:
            cv2.imshow("Classroom Video Analysis", frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break

        frame_idx += 1

    cap.release()
    writer.release()
    if show:
        cv2.destroyAllWindows()

    detail_fields = [
        "timestamp", "student", "state", "ear", "mar", "yaw", "pitch",
        "roll", "attention_score", "phone_detected", "patterns",
    ]
    write_csv(detail_csv, detail_rows, detail_fields)

    summary_rows = [students[track_id].summary() for track_id in sorted(students)]
    summary_fields = [
        "student", "total_seconds", "final_attention_score", "ok_seconds",
        "distracted_seconds", "sleeping_seconds", "talking_seconds",
        "phone_usage_seconds", "body_only_seconds", "absent_seconds", "pattern_alerts",
    ]
    write_csv(summary_csv, summary_rows, summary_fields)

    elapsed = time.time() - start
    return {
        "output_video": output_video,
        "detail_csv": detail_csv,
        "summary_csv": summary_csv,
        "students": len(students),
        "frames": frame_idx,
        "elapsed_seconds": elapsed,
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze a recorded classroom video.")
    parser.add_argument("video", nargs="?", help="Path to input classroom video")
    parser.add_argument("--show", action="store_true", help="Show annotated video while processing")
    args = parser.parse_args()

    video = args.video or input("Enter video path: ").strip().strip('"')
    result = analyze_video(Path(video), show=args.show)
    print("[INFO] Video analysis completed.")
    print(f"[INFO] Students tracked: {result['students']}")
    print(f"[INFO] Frames processed: {result['frames']}")
    print(f"[INFO] Output video: {result['output_video']}")
    print(f"[INFO] Details CSV: {result['detail_csv']}")
    print(f"[INFO] Summary CSV: {result['summary_csv']}")


if __name__ == "__main__":
    main()
