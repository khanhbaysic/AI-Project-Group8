from collections import defaultdict

from src.behavior_analyzer.attention_score import AttentionScorer
from src.behavior_analyzer.pattern_detector import PatternDetector
from src.behavior_analyzer.temporal_buffer import TemporalBuffer
from src.eye_monitor import EyeMonitor
from src.liveness_detector import LivenessDetector
from src.mouth_monitor import MouthMonitor
from src.states import ABSENT, BODY_ONLY, DISTRACTED, OK, PHONE_USAGE, SLEEPING, SPOOFING, TALKING


class StudentState:
    def __init__(self, student_label, config):
        self.student_label = student_label
        self.eye_monitor = EyeMonitor(
            config["ear_threshold"],
            config["sleep_duration"],
            config.get("sleep_use_min_eye_ear", True),
            config.get("sleep_min_eye_threshold_factor", 0.9),
        )
        self.mouth_monitor = MouthMonitor(
            config["mar_threshold"], config["talk_duration"],
            talk_window=config.get("talk_window", 1.5),
            talk_min_transitions=config.get("talk_min_transitions", 3),
            talk_mar_variance_threshold=config.get("talk_mar_variance_threshold", 0.005),
        )
        # Anti-spoofing (liveness) per tracked student. This needs NO student ID
        # and NO reference image -- it only inspects the face landmarks over
        # time -- so, unlike identity, it works in the multi-student video mode.
        self.liveness_detector = LivenessDetector(
            config["liveness_threshold"],
            config["liveness_warmup_seconds"],
            config["liveness_window_seconds"],
            config["ear_threshold"],
            config.get("liveness_spoof_confirm_seconds", 4.0),
            config.get("liveness_require_blink", True),
            config.get("liveness_max_score_without_blink", 0.29),
            config.get("liveness_no_blink_grace_seconds", 15.0),
        )
        self.last_liveness_status = "NO_FACE"
        self.last_liveness_score = 0.0
        self.attention = AttentionScorer(config["attention_rates"], config["attention_alpha"])
        self.buffer = TemporalBuffer(config["buffer_seconds"])
        self.pattern_detector = PatternDetector(
            config.get("progressive_drowsiness_ear_threshold", config["ear_threshold"]),
            config.get("frequent_distraction_window", 30.0),
            config.get("frequent_distraction_seconds", 3.0),
            config.get("sustained_distraction_seconds", 5.0),
            config.get("rapid_attention_drop_window", 20.0),
            config.get("rapid_attention_drop_points", 30.0),
        )
        self.state_durations = defaultdict(float)
        self.alert_counts = defaultdict(int)
        self.active_patterns = set()
        self.last_record = None

    def update(self, record, dt):
        # carry the latest liveness verdict into every record (refreshed by
        # check_liveness when a face is present; carried over otherwise)
        record["liveness_status"] = self.last_liveness_status
        # When anti-spoofing fires, override the behavioral state so that
        # the `state` column in the details CSV reflects SPOOFING — this
        # makes it visible to the evaluation harness and the heatmap.
        if self.last_liveness_status == SPOOFING:
            record["state"] = SPOOFING
        state = record["state"]
        _, display_score = self.attention.update(state, dt)
        record["attention_score"] = display_score
        self.state_durations[state] += dt
        self.buffer.add(record.copy())
        patterns = self.pattern_detector.detect(self.buffer)
        current_patterns = set(patterns)
        for pattern in current_patterns - self.active_patterns:
            self.alert_counts[pattern] += 1
        self.active_patterns = current_patterns
        self.last_record = record.copy()
        return display_score, patterns

    def check_liveness(self, now, landmarks, ear, yaw, pitch):
        """Update this student's liveness detector from the current face and
        return the status ('CHECKING' / 'LIVE' / 'SPOOFING').

        Identity is NOT checked here (that needs an enrolled ID + reference
        image). Liveness only needs the face landmarks over time, so unlike
        identity it can run for every tracked student in the video analyzer.
        """
        score, status = self.liveness_detector.update(now, landmarks, ear, yaw, pitch)
        self.last_liveness_score = score
        self.last_liveness_status = status
        return status

    def state_contributions(self):
        """Return per-state duration and approximate score impact.

        The impact is computed as ``seconds * rate`` using the same rate table
        as the attention scorer. It is an explanation aid for the report, not a
        second scoring system.
        """
        rows = []
        for state, seconds in sorted(self.state_durations.items()):
            rate = self.attention.rates.get(state, 0.0)
            rows.append({
                "state": state,
                "seconds": round(seconds, 2),
                "rate": rate,
                "impact": round(seconds * rate, 2),
            })
        return rows

    def summary(self):
        total = sum(self.state_durations.values())
        contributions = self.state_contributions()
        return {
            "student": self.student_label,
            "total_seconds": round(total, 2),
            "final_attention_score": round(self.attention.display_score, 2),
            "ok_seconds": round(self.state_durations[OK], 2),
            "distracted_seconds": round(self.state_durations[DISTRACTED], 2),
            "sleeping_seconds": round(self.state_durations[SLEEPING], 2),
            "talking_seconds": round(self.state_durations[TALKING], 2),
            "phone_usage_seconds": round(self.state_durations[PHONE_USAGE], 2),
            "body_only_seconds": round(self.state_durations[BODY_ONLY], 2),
            "absent_seconds": round(self.state_durations[ABSENT], 2),
            "score_impact": "; ".join(
                f"{row['state']}:{row['impact']:+.2f}"
                for row in contributions
            ),
            "pattern_alerts": "; ".join(f"{k}:{v}" for k, v in sorted(self.alert_counts.items())),
        }
