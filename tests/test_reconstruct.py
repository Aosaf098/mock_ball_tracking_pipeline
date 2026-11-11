
import unittest
from src.reconstruct import reconstruct_trajectory


def make_event(t, status, x=None, y=None):
    return {
        "timestamp": t,
        "status": status,
        "position": {"x": x, "y": y},
        "source": "simulated_camera_01",
        "last_observed_time": t if status == "observed" else None,
        "confidence": None,
        "uncertainty": None
    }


class TestReconstruction(unittest.TestCase):

    def test_linear_interpolation(self):
        events = [
            make_event(0.0, "observed", 0, 0),
            make_event(0.1, "missing"),
            make_event(0.2, "observed", 2, 4)
        ]

        result, filled = reconstruct_trajectory(events)

        self.assertEqual(filled, 1)
        self.assertEqual(
            result[1]["position"],
            {"x": 1.0, "y": 2.0}
        )
        self.assertEqual(result[1]["status"], "predicted")
        self.assertEqual(result[1]["last_observed_time"], 0.0)

    def test_boundary_gaps_remain_missing(self):
        events = [
            make_event(0.0, "missing"),
            make_event(0.1, "observed", 1, 1),
            make_event(0.2, "missing")
        ]

        result, filled = reconstruct_trajectory(events)

        self.assertEqual(filled, 0)
        self.assertEqual(result[0]["status"], "missing")
        self.assertEqual(result[2]["status"], "missing")

    def test_long_gap_not_reconstructed(self):
        events = [
            make_event(0.0, "observed", 0, 0),
            make_event(0.1, "missing"),
            make_event(0.2, "missing"),
            make_event(0.3, "missing"),
            make_event(0.4, "observed", 4, 4)
        ]

        result, filled = reconstruct_trajectory(
            events, max_gap_frames=2
        )

        self.assertEqual(filled, 0)
        self.assertEqual(result[2]["status"], "missing")

    def test_original_data_not_modified(self):
        events = [
            make_event(0.0, "observed", 0, 0),
            make_event(0.1, "missing"),
            make_event(0.2, "observed", 2, 2)
        ]

        result, _ = reconstruct_trajectory(events)

        self.assertEqual(events[1]["status"], "missing")
        self.assertEqual(result[1]["status"], "predicted")


if __name__ == "__main__":
    unittest.main()
