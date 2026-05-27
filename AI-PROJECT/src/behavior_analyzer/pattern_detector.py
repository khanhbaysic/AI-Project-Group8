class PatternDetector:
    def __init__(self, progressive_drowsiness_ear_threshold=0.22):
        self.progressive_drowsiness_ear_threshold = progressive_drowsiness_ear_threshold

    def detect(self, buffer):
        patterns = []
        if buffer.count_state("DISTRACTED", 30.0) > 10:
            patterns.append("Frequent Distraction")
        if buffer.continuous_duration("DISTRACTED") > 5.0:
            patterns.append("Sustained Distraction")
        if buffer.mean("ear", 3.0) < self.progressive_drowsiness_ear_threshold and buffer.slope("pitch", 5.0) < -0.5:
            patterns.append("Progressive Drowsiness")
        if buffer.score_drop(20.0) > 30.0:
            patterns.append("Rapid Attention Drop")
        return patterns
