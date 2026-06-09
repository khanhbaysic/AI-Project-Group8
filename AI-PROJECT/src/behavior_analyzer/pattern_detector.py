from src.states import DISTRACTED


class PatternDetector:
    def __init__(
        self,
        progressive_drowsiness_ear_threshold=0.22,
        frequent_distraction_window=30.0,
        frequent_distraction_seconds=3.0,
        sustained_distraction_seconds=5.0,
        rapid_attention_drop_window=20.0,
        rapid_attention_drop_points=30.0,
    ):
        self.progressive_drowsiness_ear_threshold = progressive_drowsiness_ear_threshold
        self.frequent_distraction_window = frequent_distraction_window
        self.frequent_distraction_seconds = frequent_distraction_seconds
        self.sustained_distraction_seconds = sustained_distraction_seconds
        self.rapid_attention_drop_window = rapid_attention_drop_window
        self.rapid_attention_drop_points = rapid_attention_drop_points

    def detect(self, buffer):
        patterns = []
        if buffer.duration_state(DISTRACTED, self.frequent_distraction_window) > self.frequent_distraction_seconds:
            patterns.append("Frequent Distraction")
        if buffer.continuous_duration(DISTRACTED) > self.sustained_distraction_seconds:
            patterns.append("Sustained Distraction")
        if buffer.mean("ear", 3.0) < self.progressive_drowsiness_ear_threshold and buffer.slope("pitch", 5.0) < -0.5:
            patterns.append("Progressive Drowsiness")
        if buffer.score_drop(self.rapid_attention_drop_window) > self.rapid_attention_drop_points:
            patterns.append("Rapid Attention Drop")
        return patterns
