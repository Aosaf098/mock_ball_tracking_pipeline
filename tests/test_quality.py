
import unittest

from src.quality import check_sequence


class TestDataQuality(unittest.TestCase):

    def setUp(self):
        self.events = [
            {"frame_id": i, "timestamp": round(i * 0.1, 3)}
            for i in range(5)
        ]

    def test_clean_data(self):
        report = check_sequence(self.events, 10)
        self.assertTrue(report["passed"])

    def test_duplicate_frame(self):
        events = self.events.copy()
        events.insert(2, self.events[1].copy())

        report = check_sequence(events, 10)

        self.assertEqual(report["duplicate_frames"], [1])
        self.assertFalse(report["passed"])

    def test_missing_frame(self):
        events = [
            event for event in self.events
            if event["frame_id"] != 2
        ]

        report = check_sequence(events, 10)

        self.assertEqual(report["missing_frames"], [2])
        self.assertFalse(report["passed"])

    def test_incorrect_timestamp(self):
        events = [event.copy() for event in self.events]
        events[2]["timestamp"] = 0.45

        report = check_sequence(events, 10)

        self.assertTrue(report["timing_mismatch_lines"])
        self.assertFalse(report["passed"])


if __name__ == "__main__":
    unittest.main()
