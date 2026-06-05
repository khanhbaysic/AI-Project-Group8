from pathlib import Path

from src.states import ABSENT, BODY_ONLY, DISTRACTED, OK, PHONE_USAGE, SLEEPING, TALKING


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONFIG = {
    "camera_id": 0,
    "frame_width": 1280,
    "frame_height": 720,
    "target_fps": 20,
    "video_detection_scale": 2.5,
    "video_face_crop_scale": 4.0,
    "video_face_detection_confidence": 0.2,
    "video_person_face_area_ratio": 0.75,
    "video_person_assisted_faces": True,
    "video_body_only_enabled": True,
    "video_yaw_threshold": 32.0,
    "video_strong_yaw_margin": 8.0,
    "video_distraction_enter_seconds": 0.25,
    "video_distraction_exit_seconds": 0.6,
    "video_distraction_rise_rate": 1.5,
    "video_distraction_fall_rate": 0.8,
    "video_ear_threshold": 0.035,
    "video_sleep_duration": 3.5,
    "video_progressive_drowsiness_ear_threshold": 0.035,
    "video_use_pitch_distraction": True,
    "video_absent_duration": 1.5,       # seconds missed before a track is declared ABSENT
    "video_absent_track_keep_seconds": 8.0, # keep missing tracks long enough to log true absence
    "phone_detection_enabled": True,
    "phone_model_path": "yolov8n.pt",
    "phone_confidence": 0.35,
    "phone_interval_frames": 5,
    "phone_near_student_scale": 1.8,
    "video_phone_duration": 0.5,
    "person_tracking_enabled": False,
    "person_min_box_area_ratio": 0.01,
    "person_phone_near_scale": 1.15,

    "yaw_threshold": 30.0,
    "pitch_down_threshold": -25.0,
    "ear_threshold": 0.22,
    "sleep_duration": 3.0,
    "mar_threshold": 0.52,              # lower = more sensitive mouth-open transition detection
    "talk_duration": 1.0,                # lower = talking is confirmed sooner
    "talk_window": 1.5,                  # rolling window (seconds) for oscillation analysis
    "talk_min_transitions": 2,           # lower = fewer open/close cycles needed for speech
    "talk_mar_variance_threshold": 0.003, # lower = subtler mouth motion can trigger speech
    "absent_duration": 3.0,

    "attention_alpha": 0.15,
    "attention_rates": {
        OK: 1.0,
        DISTRACTED: -2.0,
        TALKING: -3.0,
        SLEEPING: -5.0,
        ABSENT: -6.0,
        PHONE_USAGE: -4.0,
        BODY_ONLY: 0.0,
    },

    "buffer_seconds": 60.0,
    "pattern_interval": 0.5,

    "liveness_threshold": 0.38,
    "liveness_warmup_seconds": 1.5,
    "liveness_window_seconds": 3.5,

    "identity_similarity_threshold": 0.88,
    "identity_check_interval": 0.5,

    "database_csv": PROJECT_ROOT / "database" / "students.csv",
    "evidence_dir": PROJECT_ROOT / "evidence",
    "evidence_images_dir": PROJECT_ROOT / "evidence" / "images",
    "violations_csv": PROJECT_ROOT / "evidence" / "violations.csv",
}
