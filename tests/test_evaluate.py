
import unittest

from src.evaluate import (
    evaluate_dataset,
    paired_comparison,
    position_errors
)


def event(frame, x, y=0.0, status="observed",
          timestamp=None):
    return {
        "match_id": "match_001",
        "ball_id": "ball_01",
        "frame_id": frame,
        "timestamp": (
            frame * 0.1 if timestamp is None else timestamp
        ),
        "coordinate_frame": "pitch_center",
        "position_unit": "meters",
        "status": status,
        "position": {
            "x": x,
            "y": y
        }
    }


class TestEvaluation(unittest.TestCase):

    def test_known_euclidean_error(self):
        result = evaluate_dataset(
            [event(0, 0)],
            [event(0, 3, 4)]
        )

        self.assertEqual(
            result["all_available"]["mae_m"], 5.0
        )
        self.assertEqual(
            result["all_available"]["rmse_m"], 5.0
        )

    def test_missing_values_reduce_coverage(self):
        truth = [
            event(0, 0),
            event(1, 1)
        ]

        measured = [
            event(0, 0),
            event(1, None, None, "missing")
        ]

        result = evaluate_dataset(truth, measured)

        self.assertEqual(result["coverage_pct"], 50.0)
        self.assertEqual(result["missing_frames"], 1)

    def test_paired_comparison_uses_common_frames(self):
        truth = [
            event(0, 0),
            event(1, 10)
        ]

        baseline = [
            event(0, None, None, "missing"),
            event(1, 12)
        ]

        candidate = [
            event(0, 5),
            event(1, 11)
        ]

        result = paired_comparison(
            truth, baseline, candidate
        )

        self.assertEqual(result["common_frames"], 1)
        self.assertEqual(result["baseline_mae_m"], 2.0)
        self.assertEqual(result["candidate_mae_m"], 1.0)
        self.assertEqual(result["mae_reduction_pct"], 50.0)

    def test_mismatched_frames_rejected(self):
        with self.assertRaises(ValueError):
            position_errors(
                [event(0, 0)],
                [event(1, 0)]
            )

    def test_mismatched_timestamps_rejected(self):
        with self.assertRaises(ValueError):
            position_errors(
                [event(0, 0)],
                [event(0, 0, timestamp=0.2)]
            )

    def test_duplicate_frames_rejected(self):
        with self.assertRaises(ValueError):
            position_errors(
                [event(0, 0)],
                [event(0, 0), event(0, 0)]
            )


if __name__ == "__main__":
    unittest.main()
