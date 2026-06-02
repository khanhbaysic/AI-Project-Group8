# Intelligent E-Proctoring System

This project implements a lightweight real-time e-proctoring prototype using webcam-based facial behavior analysis.

## Features

- Real-time webcam monitoring with OpenCV.
- MediaPipe FaceMesh landmark extraction.
- Head pose estimation with `cv2.solvePnP`.
- EAR-based drowsiness detection.
- MAR-based talking detection.
- Liveness score from blink, head motion, and landmark motion cues.
- Student ID lookup from a local CSV database.
- Face verification with a lightweight FaceMesh landmark embedding.
- Continuous Attention Score from 0 to 100.
- 60-second temporal buffer and temporal pattern alerts.
- Evidence image capture and CSV logging.
- Live dashboard with state, alerts, identity, liveness, metrics, and attention trend.

## Project Structure

```text
AI-PROJECT/
  main.py
  src/
    main.py
    face_detector.py
    liveness_detector.py
    identity_verifier.py
    database.py
    head_pose.py
    eye_monitor.py
    mouth_monitor.py
    state_classifier.py
    alert_system.py
    dashboard.py
    behavior_analyzer/
      temporal_buffer.py
      attention_score.py
      pattern_detector.py
      decision_engine.py
  database/
    students.csv
    reference_images/
  evidence/
    violations.csv
    images/
  requirements.txt
  run_main.bat
```

## Setup

Install dependencies with Python 3.12:

```bat
C:\Users\PC\AppData\Local\Programs\Python\Python312\python.exe -m pip install -r requirements.txt
```

## Student Database Format

Edit `database/students.csv`:

```csv
student_id,name,reference_image
202417140,Sample Student,reference_images/202417140.jpg
```

The `reference_image` path is relative to the `database/` folder unless an absolute path is provided.

## Run

```bat
run_main.bat
```

Or:

```bat
C:\Users\PC\AppData\Local\Programs\Python\Python312\python.exe main.py
```

Enter the student ID when prompted.

## Analyze A Recorded Classroom Video

You can also use a recorded classroom video instead of the webcam workflow.

```bat
run_video_analysis.bat "C:\path\to\class_recording.mp4"
```

Or:

```bat
C:\Users\PC\AppData\Local\Programs\Python\Python312\python.exe -m src.video_analyzer "C:\path\to\class_recording.mp4"
```

The analyzer automatically detects and tracks faces as:

```text
Student_1
Student_2
Student_3
```

Outputs are saved to:

```text
output/video_analysis/
```

Generated files:

```text
<video_name>_annotated.mp4
<video_name>_details.csv
<video_name>_summary.csv
```

Current limitation: the first version uses centroid tracking, so student IDs are stable when students remain roughly in place. If students cross over each other or leave/re-enter often, a stronger tracker such as DeepSORT or face embeddings should be added.

For classroom video analysis, downward head pose is not treated as distracted by default because students may be writing notes or doing a paper exercise. The video mode mainly flags strong side-looking, sleeping, talking, and absence. You can re-enable downward distraction by setting `video_use_pitch_distraction` to `True` in `src/config.py`.

### Optional Phone Usage Detection

The video analyzer includes optional phone detection. It uses YOLO through the `ultralytics` package to detect the COCO class `cell phone`.

Install the optional dependency:

```bat
C:\Users\PC\AppData\Local\Programs\Python\Python312\python.exe -m pip install ultralytics
```

Then run the analyzer again. The first run may download `yolov8n.pt` if it is not already available.

Relevant settings in `src/config.py`:

```python
"phone_detection_enabled": True,
"phone_model_path": "yolov8n.pt",
"phone_confidence": 0.35,
"phone_interval_frames": 5,
"phone_near_student_scale": 1.8,
```

When a phone is detected near a tracked student, that student is labeled:

```text
PHONE_USAGE
```

and `phone_usage_seconds` appears in the summary CSV.

## How To Test

- `ABSENT`: leave the camera frame.
- `DISTRACTED`: look left/right or downward beyond the configured thresholds.
- `SLEEPING`: close eyes for at least 3 seconds.
- `TALKING`: keep mouth open above the MAR threshold for at least 2 seconds.
- `SPOOFING`: hold a static image/photo in front of the webcam; after warmup the liveness score should remain low.
- `Identity Mismatch`: use a different face from the reference image in `database/students.csv`.
- `Attention Score`: observe the score decrease during negative states and recover slowly during `OK`.

## Notes And Limitations

The current identity verifier uses a CPU-friendly FaceMesh landmark embedding and cosine similarity. This is explainable and easy to run, but it is not as accurate as a production face recognition model. For stronger identity verification, replace `src/identity_verifier.py` with ArcFace, InsightFace, DeepFace, or another face embedding model.

The liveness detector uses blink and motion cues. It is suitable for an academic prototype, but a dedicated anti-spoofing model would be required for high-stakes deployment.
