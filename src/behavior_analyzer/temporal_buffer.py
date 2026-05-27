from collections import deque

import numpy as np


class TemporalBuffer:
    def __init__(self, window_seconds=60.0):
        self.window_seconds = window_seconds
        self.records = deque()

    def add(self, record):
        self.records.append(record)
        now = record["timestamp"]
        while self.records and now - self.records[0]["timestamp"] > self.window_seconds:
            self.records.popleft()

    def last(self, seconds):
        if not self.records:
            return []
        now = self.records[-1]["timestamp"]
        return [r for r in self.records if now - r["timestamp"] <= seconds]

    def count_state(self, state, seconds):
        return sum(1 for r in self.last(seconds) if r.get("state") == state)

    def continuous_duration(self, state):
        if not self.records:
            return 0.0
        end = self.records[-1]["timestamp"]
        start = end
        for record in reversed(self.records):
            if record.get("state") != state:
                break
            start = record["timestamp"]
        return end - start

    def mean(self, field, seconds):
        values = [r.get(field) for r in self.last(seconds) if r.get(field) is not None]
        return float(np.mean(values)) if values else 0.0

    def slope(self, field, seconds):
        records = self.last(seconds)
        values = [(r["timestamp"], r.get(field)) for r in records if r.get(field) is not None]
        if len(values) < 2:
            return 0.0
        t = np.array([x[0] for x in values], dtype=np.float32)
        y = np.array([x[1] for x in values], dtype=np.float32)
        t = t - t[0]
        if float(np.std(t)) <= 1e-6:
            return 0.0
        return float(np.polyfit(t, y, 1)[0])

    def score_drop(self, seconds):
        records = self.last(seconds)
        scores = [r.get("attention_score") for r in records if r.get("attention_score") is not None]
        if len(scores) < 2:
            return 0.0
        return float(scores[0] - scores[-1])
