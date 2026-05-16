class StateClassifier:
    def __init__(self, yaw_threshold=30.0, pitch_down_threshold=-25.0):
        self.yaw_threshold = yaw_threshold
        self.pitch_down_threshold = pitch_down_threshold

    def classify(self, face_present, yaw, pitch, sleeping, talking):
        if not face_present:
            return "ABSENT"
        if sleeping:
            return "SLEEPING"
        if talking:
            return "TALKING"
        if abs(yaw) > self.yaw_threshold or pitch < self.pitch_down_threshold:
            return "DISTRACTED"
        return "OK"
