import argparse
import csv
import time
from pathlib import Path

import cv2

from src.config import CONFIG, PROJECT_ROOT
from src.face_detector import FaceDetector
from src.head_pose import HeadPoseEstimator
from src.phone_detector import PhoneDetector, expanded_intersection, bbox_iou
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


def filter_people(people, width, height, min_area_ratio, nms_iou=0.45):
    min_area = width * height * min_area_ratio
    candidates = [person for person in people if bbox_area(person.bbox) >= min_area]
    candidates.sort(key=lambda person: (person.confidence, bbox_area(person.bbox)), reverse=True)

    kept = []
    for person in candidates:
        if any(bbox_iou(person.bbox, existing.bbox) >= nms_iou for existing in kept):
            continue
        kept.append(person)
    return kept


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


def frame_distance_ratio(point_a, point_b, width, height):
    diagonal = max((width ** 2 + height ** 2) ** 0.5, 1.0)
    return ((point_a[0] - point_b[0]) ** 2 + (point_a[1] - point_b[1]) ** 2) ** 0.5 / diagonal


def near_recent_phone(bbox, phone_centers, width, height, max_ratio):
    if not phone_centers:
        return False
    center = bbox_center(bbox)
    nearest = min(frame_distance_ratio(center, phone_center, width, height) for phone_center in phone_centers)
    return nearest <= max_ratio


def phone_center_is_owned_by_person(
    phone_bbox,
    person_bbox,
    max_relative_y,
    min_width_ratio=0.0,
    min_height_ratio=0.0,
):
    phone_center = bbox_center(phone_bbox)
    if not point_inside_bbox(phone_center, person_bbox):
        return False

    _, py, pw, ph = person_bbox
    _, _, phone_w, phone_h = phone_bbox
    relative_y = (phone_center[1] - py) / max(ph, 1)
    width_ratio = phone_w / max(pw, 1)
    height_ratio = phone_h / max(ph, 1)
    return (
        relative_y <= max_relative_y
        and width_ratio >= min_width_ratio
        and height_ratio >= min_height_ratio
    )


def assign_phones_to_people(
    person_bboxes,
    phone_detections,
    max_relative_y=0.92,
    min_width_ratio=0.0,
    min_height_ratio=0.0,
):
    owners = set()
    assigned_phone_indices = set()
    for phone_idx, phone in enumerate(phone_detections):
        candidates = [
            (normalized_distance_to_bbox_center(bbox_center(phone.bbox), bbox), idx)
            for idx, bbox in enumerate(person_bboxes)
            if phone_center_is_owned_by_person(
                phone.bbox,
                bbox,
                max_relative_y,
                min_width_ratio,
                min_height_ratio,
            )
        ]
        if candidates:
            owners.add(min(candidates)[1])
            assigned_phone_indices.add(phone_idx)
    return owners, assigned_phone_indices


def analyze_video(video_path: Path, show=False, output_dir=OUTPUT_DIR, labels_csv=None):
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    output_dir = Path(output_dir)
    if output_dir.name != video_path.stem:
        output_dir = output_dir / video_path.stem
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
        min_detection_confidence=CONFIG.get("video_face_min_detection_confidence", 0.5),
        min_tracking_confidence=CONFIG.get("video_face_min_tracking_confidence", 0.5),
    )
    face_tracker = CentroidTracker(max_distance=max(width, height) * 0.08, max_missed=int(fps * 2))
    person_tracker = CentroidTracker(max_distance=max(width, height) * 0.12, max_missed=int(fps * 2))
    head_pose = HeadPoseEstimator()
    pitch_threshold = CONFIG["pitch_down_threshold"] if CONFIG.get("video_use_pitch_distraction", False) else -999.0
    classifier = StateClassifier(CONFIG.get("video_yaw_threshold", CONFIG["yaw_threshold"]), pitch_threshold)
    video_config = dict(CONFIG)
    video_config["ear_threshold"] = CONFIG.get("video_ear_threshold", CONFIG["ear_threshold"])
    video_config["sleep_duration"] = CONFIG.get("video_sleep_duration", CONFIG["sleep_duration"])
    video_config["mar_threshold"] = CONFIG.get("video_mar_threshold", CONFIG["mar_threshold"])
    video_config["talk_duration"] = CONFIG.get("video_talk_duration", CONFIG["talk_duration"])
    video_config["progressive_drowsiness_ear_threshold"] = CONFIG.get(
        "video_progressive_drowsiness_ear_threshold",
        video_config["ear_threshold"],
    )
    phone_detector = PhoneDetector(
        CONFIG.get("phone_model_path", "yolov8n.pt"),
        CONFIG.get("phone_confidence", 0.35),
        CONFIG.get("phone_detection_enabled", True),
        person_confidence=CONFIG.get("person_confidence", CONFIG.get("phone_confidence", 0.35)),
        phone_min_aspect_ratio=CONFIG.get("phone_min_aspect_ratio", 1.2),
        phone_max_aspect_ratio=CONFIG.get("phone_max_aspect_ratio", 4.8),
    )
    if phone_detector.warning:
        print(f"[WARN] {phone_detector.warning}")
    phone_detections = []
    person_detections = []
    students: dict[int, StudentState] = {}
    distraction_started_at: dict[int, float] = {}
    phone_started_at: dict[int, float] = {}
    phone_last_seen_at: dict[int, float] = {}
    peer_distraction_started_at: dict[int, float] = {}
    confirmed_phone_context_until = 0.0
    confirmed_phone_centers = []
    person_memory: dict[int, dict] = {}
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
                phone_detections, person_detections = phone_detector.detect_people_and_phones(
                    frame,
                    track_people=CONFIG.get("person_tracker_backend", "centroid") == "yolo",
                )
                person_detections = filter_people(
                    person_detections,
                    width,
                    height,
                    CONFIG.get("person_min_box_area_ratio", 0.01),
                    CONFIG.get("person_nms_iou", 0.45),
                )
            else:
                phone_detections = phone_detector.detect(frame)
                person_detections = []

        active_phone_users = 0
        active_phone_centers = []
        memory_visible_count = 0
        assigned_phone_indices = set()

        use_yolo_backend = CONFIG.get("person_tracker_backend", "centroid") == "yolo"
        use_person_tracking = (
            CONFIG.get("person_tracking_enabled", True)
            and phone_detector.available
            and (bool(person_detections) or (use_yolo_backend and bool(person_memory)))
        )

        if use_person_tracking:
            person_bboxes = [person.bbox for person in person_detections]
            phone_owner_indices, assigned_phone_indices = assign_phones_to_people(
                person_bboxes,
                phone_detections,
                CONFIG.get("phone_owner_max_relative_y", 0.92),
                CONFIG.get("phone_owner_min_width_ratio", 0.0),
                CONFIG.get("phone_owner_min_height_ratio", 0.0),
            )
            use_yolo_track_ids = (
                use_yolo_backend
                and (not person_detections or any(person.track_id is not None for person in person_detections))
            )
            if use_yolo_track_ids:
                assignments = {}
                assigned_track_ids = set()
                assigned_bboxes = []
                ordered_detection_indices = sorted(
                    range(len(person_detections)),
                    key=lambda idx: (
                        person_detections[idx].track_id not in person_memory,
                        -person_detections[idx].confidence,
                    ),
                )

                for det_idx in ordered_detection_indices:
                    person = person_detections[det_idx]
                    proposed_track_id = person.track_id
                    chosen_track_id = None

                    if proposed_track_id in person_memory and proposed_track_id not in assigned_track_ids:
                        chosen_track_id = proposed_track_id

                    available_memory_ids = set(person_memory) - assigned_track_ids
                    if chosen_track_id is None and available_memory_ids:
                        candidates = []
                        person_center = bbox_center(person.bbox)
                        for memory_track_id in available_memory_ids:
                            memory_bbox = person_memory[memory_track_id]["bbox"]
                            overlap = bbox_iou(person.bbox, memory_bbox)
                            distance = frame_distance_ratio(
                                person_center,
                                bbox_center(memory_bbox),
                                width,
                                height,
                            )
                            if overlap >= 0.08 or distance <= 0.16:
                                candidates.append((overlap, -distance, memory_track_id))

                        if candidates:
                            _, _, chosen_track_id = max(candidates)

                    duplicate_existing_box = any(
                        bbox_iou(person.bbox, assigned_bbox) >= 0.45
                        for assigned_bbox in assigned_bboxes
                    )
                    allow_new_track = (
                        timestamp <= CONFIG.get("person_new_track_grace_seconds", 2.0)
                        or not students
                        or proposed_track_id in students
                    )
                    if (
                        chosen_track_id is None
                        and proposed_track_id is not None
                        and not duplicate_existing_box
                        and allow_new_track
                    ):
                        chosen_track_id = proposed_track_id

                    if chosen_track_id is None:
                        continue

                    assignments[det_idx] = chosen_track_id
                    assigned_track_ids.add(chosen_track_id)
                    assigned_bboxes.append(person.bbox)

                tracks = {}
                for det_idx, track_id in assignments.items():
                    person_memory[track_id] = {
                        "bbox": person_bboxes[det_idx],
                        "last_seen": timestamp,
                    }
            else:
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
                    phone_last_seen_at[track_id] = timestamp
                else:
                    last_seen = phone_last_seen_at.get(track_id)
                    if last_seen is None or timestamp - last_seen > CONFIG.get("video_phone_hold_seconds", 0.0):
                        phone_started_at.pop(track_id, None)
                        phone_last_seen_at.pop(track_id, None)

                phone_detected = (
                    track_id in phone_started_at
                    and timestamp - phone_started_at[track_id] >= CONFIG.get("video_phone_duration", 0.0)
                    and timestamp - phone_last_seen_at.get(track_id, timestamp) <= CONFIG.get("video_phone_hold_seconds", 0.0)
                )

                if face_idx is not None:
                    used_face_indices.add(face_idx)
                    landmarks = detection.landmarks_list[face_idx]
                    face_bbox = detection.faces[face_idx]
                    yaw, pitch, roll = head_pose.estimate(frame, landmarks)
                    ear, sleeping = students[track_id].eye_monitor.check(landmarks, timestamp)
                    mar, talking = students[track_id].mouth_monitor.check(landmarks, timestamp)
                    students[track_id].check_liveness(timestamp, landmarks, ear, yaw, pitch)
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
                    active_phone_users += 1
                    active_phone_centers.append(bbox_center(bbox))
                    confirmed_phone_context_until = timestamp + CONFIG.get("video_phone_context_seconds", 2.5)
                    peer_distraction_started_at.pop(track_id, None)
                elif CONFIG.get("video_peer_distraction_enabled", True) and timestamp <= confirmed_phone_context_until:
                    peer_candidate = False
                    if face_idx is not None and abs(yaw) >= CONFIG.get("video_peer_yaw_threshold", 35.0):
                        peer_candidate = True
                    elif (
                        state == "BODY_ONLY"
                        and CONFIG.get("video_peer_body_only_enabled", True)
                        and near_recent_phone(
                            bbox,
                            confirmed_phone_centers,
                            width,
                            height,
                            CONFIG.get("video_peer_body_distance_ratio", 0.45),
                        )
                    ):
                        peer_candidate = True

                    if peer_candidate:
                        peer_distraction_started_at.setdefault(track_id, timestamp)
                        if timestamp - peer_distraction_started_at[track_id] >= CONFIG.get("video_peer_distraction_duration", 0.35):
                            state = "DISTRACTED"
                    else:
                        peer_distraction_started_at.pop(track_id, None)
                else:
                    peer_distraction_started_at.pop(track_id, None)

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

            if use_yolo_track_ids:
                memory_seconds = CONFIG.get("person_memory_seconds", 0.0)
                for track_id, memory in list(person_memory.items()):
                    if track_id in matched_track_ids:
                        continue

                    missing_for = timestamp - memory["last_seen"]
                    if missing_for > memory_seconds:
                        del person_memory[track_id]
                        continue

                    label = f"Student_{track_id}"
                    if track_id not in students:
                        students[track_id] = StudentState(label, video_config)

                    state = "BODY_ONLY"
                    record = {
                        "timestamp": timestamp,
                        "student": label,
                        "state": state,
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
                    memory_visible_count += 1

                    x, y, w, h = memory["bbox"]
                    color = color_for_state(state)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 1)
                    cv2.putText(frame, f"{label} {state} {score:.1f}", (x, max(20, y - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
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
                students[track_id].check_liveness(timestamp, landmarks, ear, yaw, pitch)
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
                    phone_last_seen_at[track_id] = timestamp
                else:
                    last_seen = phone_last_seen_at.get(track_id)
                    if last_seen is None or timestamp - last_seen > CONFIG.get("video_phone_hold_seconds", 0.0):
                        phone_started_at.pop(track_id, None)
                        phone_last_seen_at.pop(track_id, None)

                phone_detected = (
                    track_id in phone_started_at
                    and timestamp - phone_started_at[track_id] >= CONFIG.get("video_phone_duration", 0.0)
                    and timestamp - phone_last_seen_at.get(track_id, timestamp) <= CONFIG.get("video_phone_hold_seconds", 0.0)
                )
                if phone_detected:
                    state = "PHONE_USAGE"
                    active_phone_users += 1
                    active_phone_centers.append(bbox_center(bbox))
                    confirmed_phone_context_until = timestamp + CONFIG.get("video_phone_context_seconds", 2.5)
                    peer_distraction_started_at.pop(track_id, None)
                elif CONFIG.get("video_peer_distraction_enabled", True) and timestamp <= confirmed_phone_context_until:
                    peer_candidate = abs(yaw) >= CONFIG.get("video_peer_yaw_threshold", 35.0)
                    if peer_candidate:
                        peer_distraction_started_at.setdefault(track_id, timestamp)
                        if timestamp - peer_distraction_started_at[track_id] >= CONFIG.get("video_peer_distraction_duration", 0.35):
                            state = "DISTRACTED"
                    else:
                        peer_distraction_started_at.pop(track_id, None)
                else:
                    peer_distraction_started_at.pop(track_id, None)

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

        if active_phone_centers:
            confirmed_phone_centers = active_phone_centers

        for phone_idx, phone in enumerate(phone_detections):
            if use_person_tracking and phone_idx not in assigned_phone_indices:
                continue
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
        cv2.putText(frame, f"Time: {timestamp:.1f}s | {visible_label}: {len(assignments) + memory_visible_count} | IDs: {len(students)} | Raw Phones: {len(phone_detections)} | Active Phone Users: {active_phone_users}",
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
        "liveness_status",
    ]
    write_csv(detail_csv, detail_rows, detail_fields)

    summary_rows = [students[track_id].summary() for track_id in sorted(students)]
    summary_fields = [
        "student", "total_seconds", "final_attention_score", "ok_seconds",
        "distracted_seconds", "sleeping_seconds", "talking_seconds",
        "phone_usage_seconds", "body_only_seconds", "absent_seconds",
        "score_impact", "pattern_alerts",
    ]
    write_csv(summary_csv, summary_rows, summary_fields)

    try:
        from src.analytics import run_post_analytics
        run_post_analytics(detail_csv, labels_csv, output_dir=output_dir)
    except Exception as exc:
        print(f"[WARN] Post-analysis failed: {exc}")

    elapsed = time.time() - start
    return {
        "output_video": output_video,
        "detail_csv": detail_csv,
        "summary_csv": summary_csv,
        "output_dir": output_dir,
        "students": len(students),
        "frames": frame_idx,
        "elapsed_seconds": elapsed,
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze a recorded classroom video.")
    parser.add_argument("video", nargs="?", help="Path to input classroom video")
    parser.add_argument("--show", action="store_true", help="Show annotated video while processing")
    parser.add_argument("--labels", default=None, help="Path to labels_segment CSV for evaluation")
    args = parser.parse_args()

    video = args.video or input("Enter video path: ").strip().strip('"')
    result = analyze_video(Path(video), show=args.show, labels_csv=args.labels)
    print("[INFO] Video analysis completed.")
    print(f"[INFO] Students tracked: {result['students']}")
    print(f"[INFO] Frames processed: {result['frames']}")
    print(f"[INFO] Output folder: {result['output_dir']}")
    print(f"[INFO] Output video: {result['output_video']}")
    print(f"[INFO] Details CSV: {result['detail_csv']}")
    print(f"[INFO] Summary CSV: {result['summary_csv']}")


if __name__ == "__main__":
    main()