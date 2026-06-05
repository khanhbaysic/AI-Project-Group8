# Intelligent E-Proctoring System

Rule-based computer-vision prototype for online proctoring and classroom
attention analysis. The system uses MediaPipe FaceMesh landmarks, OpenCV,
temporal rules, an attention score, optional YOLOv8 phone/person detection,
and privacy-first session reports.

## Important Notes

- This project is rule-based. It does not train a custom AI model.
- `yolov8n.pt` is a pre-trained YOLO COCO model used only for `person` and
  `cell phone` detection.
- Face landmarks come from MediaPipe FaceMesh, not Haar Cascade.
- Use Python 3.12 on Windows. Python 3.14 is not supported by MediaPipe at the
  time this project was configured.

## Project Structure

```text
AI-PROJECT/
  main.py                         # webcam entry point wrapper
  run_main.bat                    # run webcam mode
  run_video_analysis.bat          # run offline video mode
  requirements.txt
  database/
    students.csv
    reference_images/
  evidence/
    violations.csv
    images/
  output/
    webcam_session/
    video_analysis/
  src/
    main.py                       # live webcam, single student
    video_analyzer.py             # offline video, multi-student
    states.py                     # canonical 7 states and shared colors
    dashboard.py
    face_detector.py
    head_pose.py
    eye_monitor.py
    mouth_monitor.py
    phone_detector.py
    identity_verifier.py
    liveness_detector.py
    behavior_analyzer/
    analytics/
  tests/
```

## Setup on Windows

Open PowerShell in the project folder:

```powershell
cd C:\Users\PC\Documents\AI-Project-Group8\AI-PROJECT
py -3.12 -m venv venv
.\venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If `requirements.txt` is not found, you are probably in the wrong folder. Run
`dir` and make sure `requirements.txt` is listed.

## Student Database Format

`database/students.csv`:

```csv
student_id,name,reference_image
202417140,Student Name,reference_images/202417140.jpg
```

Reference image paths are relative to `database/`.

## Run Webcam Mode

```powershell
python main.py
```

Then enter a student ID from `database/students.csv`.

Controls:

- Press `q` or `ESC` to exit.
- Closing the OpenCV window also stops the system.

Webcam output:

- Session CSV: `output/webcam_session/*_details.csv`
- Evidence images and alerts: `evidence/`
- Session report: `analysis_statistics/session_report.html`

## Run Offline Video Analysis

Put an `.mp4` file anywhere convenient, for example:

```text
videos/classroom_test.mp4
```

Run:

```powershell
python -m src.video_analyzer videos\classroom_test.mp4
```

Video output:

- Annotated video: `output/video_analysis/classroom_test_annotated.mp4`
- Details CSV: `output/video_analysis/classroom_test_details.csv`
- Summary CSV: `output/video_analysis/classroom_test_summary.csv`

## Generate Session Report

```powershell
python -m src.analytics.session_report output\video_analysis\classroom_test_details.csv
```

The report is privacy-first: it uses geometric and aggregate data only, not
saved face images.

## Evaluation Workflow

Create a label template from a details CSV:

```powershell
python -m src.analytics.make_label_template output\video_analysis\classroom_test_details.csv --output labels_segment.csv
```

Fill `true_state` manually using one of:

```text
OK, DISTRACTED, TALKING, SLEEPING, ABSENT, PHONE_USAGE, BODY_ONLY
```

Then run:

```powershell
python -m src.analytics.evaluate --mode segment --labels labels_segment.csv --details output\video_analysis\classroom_test_details.csv --out-dir analysis_statistics
```

The evaluation report prints all 7 states. If a state has no ground-truth
labels, it appears as `NO LABELS` instead of being silently skipped.

## Run Tests

```powershell
python -m unittest discover -s tests
```

The tests cover pure-logic pieces that do not need a webcam:

- EAR calculation
- MAR/talking oscillation logic
- Attention score movement
- Centroid tracker ID stability

## Current Behavioral States

- `OK`: focused
- `DISTRACTED`: looking away or down
- `TALKING`: mouth oscillation detected
- `SLEEPING`: eyes closed too long
- `ABSENT`: tracked student missing long enough
- `PHONE_USAGE`: phone detected near a student
- `BODY_ONLY`: person visible but face unclear

## Known Limitations

- Identity verification currently uses lightweight landmark geometry, not
  ArcFace/InsightFace embeddings.
- Liveness is heuristic and webcam-only; it is not robust against advanced
  replay/deepfake attacks.
- Phone detection uses generic YOLO COCO classes, so fine-tuning would improve
  phone-in-hand accuracy in classroom videos.
