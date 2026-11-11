
import unittest

from src.analytics import calculate_kinematics


def make_event(frame_id, x, y=0.0, status="observed"):
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
            "y": y
        }
    }


class TestAnalytics(unittest.TestCase):

    def test_constant_velocity(self):
        events = [
            make_event(i, float(i))
            for i in range(4)
        ]

        result, summary = calculate_kinematics(events)

        self.assertAlmostEqual(
            summary["total_distance_m"], 3.0
        )
        self.assertAlmostEqual(
            summary["average_speed_m_s"], 10.0
        )
        self.assertEqual(
            result[3]["kinematics"]["acceleration_m_s2"],
            0.0
        )

    def test_acceleration(self):
        events = [
            make_event(0, 0.0),
            make_event(1, 0.1),
            make_event(2, 0.4),
            make_event(3, 0.9)
        ]

        result, summary = calculate_kinematics(events)

        self.assertAlmostEqual(
            result[2]["kinematics"]["acceleration_m_s2"],
            20.0
        )
        self.assertAlmostEqual(
            summary["maximum_acceleration_m_s2"],
            20.0
        )

    def test_missing_position_breaks_interval(self):
        events = [
            make_event(0, 0.0),
            make_event(1, None, None, "missing"),
            make_event(2, 2.0),
            make_event(3, 3.0)
        ]

        result, summary = calculate_kinematics(events)

        self.assertEqual(
            summary["total_distance_m"], 1.0
        )
        self.assertIsNone(
            result[2]["kinematics"]["speed_m_s"]
        )

    def test_predicted_position_is_marked(self):
        events = [
            make_event(0, 0.0),
            make_event(1, 1.0, status="predicted"),
            make_event(2, 2.0)
        ]

        result, summary = calculate_kinematics(events)

        self.assertEqual(summary["predicted_events"], 1)
        self.assertEqual(summary["total_distance_m"], 2.0)
        self.assertTrue(
            result[1]["kinematics"]["contains_prediction"]
        )
        self.assertTrue(
            result[2]["kinematics"]["contains_prediction"]
        )

    def test_original_data_unchanged(self):
        events = [
            make_event(0, 0.0),
            make_event(1, 1.0)
        ]

        result, _ = calculate_kinematics(events)

        self.assertNotIn("kinematics", events[1])
        self.assertIn("kinematics", result[1])


if __name__ == "__main__":
    unittest.main()
