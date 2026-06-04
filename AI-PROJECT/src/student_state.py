from collections import defaultdict

from src.behavior_analyzer.attention_score import AttentionScorer
from src.behavior_analyzer.pattern_detector import PatternDetector
from src.behavior_analyzer.temporal_buffer import TemporalBuffer
from src.eye_monitor import EyeMonitor
from src.mouth_monitor import MouthMonitor
from src.states import ABSENT, BODY_ONLY, DISTRACTED, OK, PHONE_USAGE, SLEEPING, TALKING


class StudentState:
    def __init__(self, student_label, config):
        self.student_label = student_label
        self.eye_monitor = EyeMonitor(config["ear_threshold"], config["sleep_duration"])
        self.mouth_monitor = MouthMonitor(
            config["mar_threshold"], config["talk_duration"],
            talk_window=config.get("talk_window", 1.5),
            talk_min_transitions=config.get("talk_min_transitions", 3),
            talk_mar_variance_threshold=config.get("talk_mar_variance_threshold", 0.005),
        )
        self.attention = AttentionScorer(config["attention_rates"], config["attention_alpha"])
        self.buffer = TemporalBuffer(config["buffer_seconds"])
        self.pattern_detector = PatternDetector(
            config.get("progressive_drowsiness_ear_threshold", config["ear_threshold"])
        )
        self.state_durations = defaultdict(float)
        self.alert_counts = defaultdict(int)
        self.active_patterns = set()
        self.last_record = None

    def update(self, record, dt):
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

    def summary(self):
        total = sum(self.state_durations.values())
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
            "pattern_alerts": "; ".join(f"{k}:{v}" for k, v in sorted(self.alert_counts.items())),
        }
