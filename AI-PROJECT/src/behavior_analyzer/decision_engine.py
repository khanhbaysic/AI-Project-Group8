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

        # Behavioral state alerts
        state = record.get("state", "")
        if state == "SLEEPING":
            alerts.append("Sleeping Detected")
            if level == "OK":
                level = "WARNING"
        elif state == "TALKING":
            alerts.append("Talking Detected")
            if level == "OK":
                level = "WARNING"
        elif state == "PHONE_USAGE":
            alerts.append("Phone Usage Detected")
            if level == "OK":
                level = "WARNING"

        for pattern in patterns:
            alerts.append(pattern)
            if level == "OK":
                level = "WARNING"

        return level, alerts
