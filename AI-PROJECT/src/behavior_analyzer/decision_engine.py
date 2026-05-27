class DecisionEngine:
    def decide(self, record, patterns):
        alerts = []
        level = "OK"

        if record.get("identity_status") == "MISMATCH":
            alerts.append("Identity Mismatch")
            level = "CRITICAL"
        if record.get("identity_status") == "UNKNOWN_ID":
            alerts.append("Unknown Student ID")
            level = "CRITICAL"
        if record.get("identity_status") == "NO_REFERENCE":
            alerts.append("Missing Reference Image")
            level = "CRITICAL"
        if record.get("liveness_status") == "SPOOFING":
            alerts.append("Spoofing Detected")
            level = "CRITICAL"
        if record.get("attention_score", 100) < 40:
            alerts.append("Attention Score Critical")
            level = "CRITICAL"
        elif record.get("attention_score", 100) < 60 and level != "CRITICAL":
            alerts.append("Attention Score Warning")
            level = "WARNING"

        for pattern in patterns:
            alerts.append(pattern)
            if level == "OK":
                level = "WARNING"

        return level, alerts
