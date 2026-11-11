
import unittest
from src.smooth import smooth_trajectory


def make_event(frame_id, x=None, status="observed"):
    return {
        "match_id": "match_001",
        "ball_id": "ball_01",
        "frame_id": frame_id,
        "timestamp": round(frame_id * 0.1, 3),
        "coordinate_frame": "pitch_center",
        "position_unit": "meters",
        "status": status,
        "position": {
            "x": x,
            "y": 0.0 if x is not None else None
        },
        "uncertainty": None
    }


class TestSmoothing(unittest.TestCase):

    def test_isolated_noise_spike(self):
        events = [
            make_event(i, x)
            for i, x in enumerate([0, 0, 9, 0, 0])
        ]

        result, count = smooth_trajectory(events)

        self.assertEqual(count, 1)
        self.assertEqual(result[2]["position"]["x"], 3.0)

    def test_constant_velocity(self):
        events = [
            make_event(i, float(i))
            for i in range(5)
        ]

        result, _ = smooth_trajectory(events)

        self.assertEqual(result[2]["position"]["x"], 2.0)

    def test_missing_data_preserved(self):
        events = [
            make_event(0, 0),
            make_event(1, 10),
            make_event(2, status="missing"),
            make_event(3, 10),
            make_event(4, 0)
        ]

        result, count = smooth_trajectory(events)

        self.assertEqual(count, 0)
        self.assertEqual(result[2]["status"], "missing")
        self.assertIsNone(result[2]["position"]["x"])
        self.assertEqual(result[1]["position"]["x"], 10)
        self.assertEqual(result[3]["position"]["x"], 10)

    def test_original_data_unchanged(self):
        events = [
            make_event(i, x)
            for i, x in enumerate([0, 0, 9, 0, 0])
        ]

        result, _ = smooth_trajectory(events)

        self.assertEqual(events[2]["position"]["x"], 9)
        self.assertEqual(result[2]["position"]["x"], 3)

    def test_even_window_rejected(self):
        with self.assertRaises(ValueError):
            smooth_trajectory([], window_size=4)


if __name__ == "__main__":
    unittest.main()
