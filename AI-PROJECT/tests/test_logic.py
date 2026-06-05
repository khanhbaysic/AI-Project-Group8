import unittest

import numpy as np

from src.behavior_analyzer.attention_score import AttentionScorer
from src.eye_monitor import LEFT_EYE, RIGHT_EYE, eye_aspect_ratio
from src.mouth_monitor import MOUTH_POINTS, MouthMonitor
from src.states import DISTRACTED, OK, SLEEPING
from src.video_tracker import CentroidTracker


def eye_points(open_height):
    return np.array([
        [0.0, 0.0],
        [1.0, open_height],
        [3.0, open_height],
        [4.0, 0.0],
        [3.0, -open_height],
        [1.0, -open_height],
    ], dtype=np.float32)


def landmarks_with_eyes(open_height):
    landmarks = np.zeros((478, 3), dtype=np.float32)
    pts = eye_points(open_height)
    landmarks[LEFT_EYE, :2] = pts
    landmarks[RIGHT_EYE, :2] = pts
    return landmarks


def landmarks_with_mar(mar):
    landmarks = np.zeros((478, 3), dtype=np.float32)
    v = 5.0 * mar
    pts = np.array([
        [0.0, 0.0],
        [10.0, 0.0],
        [3.0, v],
        [7.0, v],
        [7.0, -v],
        [3.0, -v],
    ], dtype=np.float32)
    landmarks[MOUTH_POINTS, :2] = pts
    return landmarks


class EyeMonitorTests(unittest.TestCase):
    def test_eye_aspect_ratio_reflects_eye_openness(self):
        open_ear = eye_aspect_ratio(eye_points(2.0))
        closed_ear = eye_aspect_ratio(eye_points(0.1))
        self.assertGreater(open_ear, closed_ear)
        self.assertAlmostEqual(open_ear, 1.0, places=3)


class MouthMonitorTests(unittest.TestCase):
    def test_talking_detects_mar_oscillation(self):
        monitor = MouthMonitor(
            mar_threshold=0.5,
            talk_duration=0.0,
            talk_window=1.0,
            talk_min_transitions=2,
            talk_mar_variance_threshold=10.0,
        )
        talking = False
        for idx, mar in enumerate([0.2, 0.8, 0.2, 0.8]):
            _, talking = monitor.check(landmarks_with_mar(mar), now=idx * 0.1)
        self.assertTrue(talking)


class AttentionScoreTests(unittest.TestCase):
    def test_attention_score_moves_with_state_rates(self):
        scorer = AttentionScorer({OK: 1.0, DISTRACTED: -2.0, SLEEPING: -5.0}, alpha=1.0)
        scorer.raw_score = 50.0
        scorer.display_score = 50.0
        scorer.update(OK, 5.0)
        self.assertEqual(scorer.raw_score, 55.0)
        scorer.update(DISTRACTED, 2.0)
        self.assertEqual(scorer.raw_score, 51.0)
        scorer.update(SLEEPING, 20.0)
        self.assertEqual(scorer.raw_score, 0.0)


class CentroidTrackerTests(unittest.TestCase):
    def test_tracker_keeps_id_for_stationary_box(self):
        tracker = CentroidTracker(max_distance=20.0, max_missed=2)
        first, _ = tracker.update([(10, 10, 20, 20)])
        second, _ = tracker.update([(12, 11, 20, 20)])
        self.assertEqual(first[0], second[0])


if __name__ == "__main__":
    unittest.main()
