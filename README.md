# Intelligent E-Proctoring System

A **rule-based** real-time e-proctoring prototype that monitors student behavior through webcam video. It uses **MediaPipe FaceMesh** for face detection and landmark extraction, hand-crafted geometric rules (EAR, MAR, head angles) for state classification, and an optional **pre-trained YOLOv8n** (`yolov8n.pt`) for person/cell-phone detection. **No custom model is trained** — all thresholds are configured in `src/config.py`.

## Features

- Real-time webcam monitoring with OpenCV.
- **MediaPipe FaceMesh** landmark extraction (478 points, no Haar Cascade).
- Head pose estimation via `cv2.solvePnP` with 14 landmarks and `atan2`-based Euler extraction.
- EAR-based drowsiness detection (eye closure ≥ 3 s).
- MAR-based talking detection (mouth open ≥ 2 s).
- Liveness score from blink rate, head motion, and landmark motion.
- Student ID lookup from a local CSV database (`database/students.csv`).
- Face verification with a lightweight FaceMesh landmark embedding.
- Continuous Attention Score (0–100) with configurable decay/recovery rates.
- 60-second temporal buffer and pattern-based alerts (e.g., repeated drowsiness).
- Evidence image capture and CSV violation logging.
- Live dashboard showing state, alerts, identity, liveness, metrics, and attention trend.
- Recorded-video analysis mode with per-student tracking and summary CSV.
- Optional phone/person detection using a **pre-trained YOLOv8n** (COCO classes `person` and `cell phone` only — no custom training).

## Project Structure

```text
AI-PROJECT/
  main.py                        # Entry point (imports src.main)
  requirements.txt
  run_main.bat
  run_video_analysis.bat
  yolov8n.pt                     # Pre-trained YOLO (person + phone detection)
  src/
    __init__.py
    config.py                    # All thresholds and settings
    main.py                      # Webcam loop + dashboard
    face_detector.py             # MediaPipe FaceMesh wrapper
    head_pose.py                 # solvePnP + atan2 Euler angles
    eye_monitor.py               # EAR calculation + sleep detection
    mouth_monitor.py             # MAR calculation + talk detection
    state_classifier.py          # Rule-based state classifier
    student_state.py             # Per-student state aggregation
    alert_system_v2.py           # Evidence capture + CSV logging
    dashboard.py                 # Live OpenCV dashboard overlay
    liveness_detector.py         # Blink/motion liveness scoring
    identity_verifier.py         # FaceMesh-embedding face verification
    database.py                  # students.csv loader
    phone_detector.py            # YOLOv8n person + phone detector
    video_analyzer.py            # Recorded-video analysis pipeline
    video_tracker.py             # Centroid-based multi-face tracker
    behavior_analyzer/
      __init__.py
      temporal_buffer.py         # Rolling metric buffer
      attention_score.py         # EMA attention score
      pattern_detector.py        # Temporal pattern rules
      decision_engine.py         # Alert decision logic
    analytics/
      __init__.py
      evaluate.py                # Accuracy evaluation helpers
      session_report.py          # Post-session HTML/PDF report
  database/
    students.csv                 # student_id, name, reference_image
    reference_images/            # One image per student
  evidence/
    violations.csv               # Auto-generated violation log
    images/                      # Auto-captured evidence screenshots
  output/
    video_analysis/              # Video analyzer outputs
```

## Setup

Install dependencies (Python 3.12):

```bat
python -m pip install -r requirements.txt
```

For phone detection, also install:

```bat
python -m pip install ultralytics
```

The first run may download `yolov8n.pt` if it is not already present.

## Student Database Format

Edit `database/students.csv`:

```csv
student_id,name,reference_image
202417140,Sample Student,reference_images/202417140.jpg
```

The `reference_image` path is relative to the `database/` folder unless an absolute path is given.

## Run (Webcam Mode)

```bat
run_main.bat
```

Or:

```bat
python main.py
```

Enter the student ID when prompted.

## Analyze a Recorded Classroom Video

```bat
run_video_analysis.bat "C:\path\to\class_recording.mp4"
```

Or:

```bat
python -m src.video_analyzer "C:\path\to\class_recording.mp4"
```

The analyzer auto-detects and tracks faces as `Student_1`, `Student_2`, etc. Outputs are saved to `output/video_analysis/`:

```text
<video_name>_annotated.mp4
<video_name>_details.csv
<video_name>_summary.csv
```

**Tracking limitation:** Centroid-based tracking works well when students stay roughly in place. For frequent crossings or re-entries, a stronger tracker (DeepSORT, face embeddings) would be needed.

### Optional Phone Usage Detection

When `ultralytics` is installed and `phone_detection_enabled` is `True` in `src/config.py`, the analyzer uses the **pre-trained YOLOv8n** to detect COCO classes `person` and `cell phone`. No custom YOLO training is performed. Relevant settings:

```python
"phone_detection_enabled": True,
"phone_model_path": "yolov8n.pt",
"phone_confidence": 0.35,
"phone_interval_frames": 5,
"phone_near_student_scale": 1.8,
```

When a phone is detected near a tracked student, that student is labeled `PHONE_USAGE`.

## How to Test

- **ABSENT**: Leave the camera frame.
- **DISTRACTED**: Look left/right or downward past the configured thresholds.
- **SLEEPING**: Close eyes for ≥ 3 seconds.
- **TALKING**: Keep mouth open above the MAR threshold for ≥ 2 seconds.
- **SPOOFING**: Hold a static photo in front of the webcam; the liveness score should stay low after warmup.
- **Identity Mismatch**: Use a different face from the one in `database/students.csv`.
- **Attention Score**: Observe it decrease during negative states and recover during `OK`.

## Notes and Limitations

- This is a **rule-based prototype** — all decisions come from hand-tuned thresholds, not a trained classifier. Accuracy depends on lighting, camera angle, and individual facial geometry.
- The identity verifier uses a CPU-friendly FaceMesh landmark embedding with cosine similarity. For stronger verification, replace with ArcFace, InsightFace, or DeepFace.
- The liveness detector uses blink and motion heuristics. A dedicated anti-spoofing model would be needed for high-stakes use.
- `yolov8n.pt` is a **pre-trained** YOLO model used only for `person` and `cell phone` detection — it is not fine-tuned on any project-specific data.
