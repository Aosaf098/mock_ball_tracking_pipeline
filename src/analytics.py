
import json
import math
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

INPUT = (
    ROOT / "data/processed/"
    "ball_tracking_match_001_smoothed.jsonl"
)

ENRICHED_OUTPUT = (
    ROOT / "data/processed/"
    "ball_tracking_match_001_enriched.jsonl"
)

SUMMARY_OUTPUT = (
    ROOT / "data/processed/"
    "ball_tracking_match_001_metrics.json"
)

VALID_STATUSES = {"observed", "predicted"}

STREAM_FIELDS = (
    "match_id",
    "ball_id",
    "coordinate_frame",
    "position_unit"
)


def can_connect(previous, current):
    """Check whether two events form a valid movement interval."""

    same_stream = all(
        previous.get(field) == current.get(field)
        for field in STREAM_FIELDS
    )

    consecutive_frames = (
        current["frame_id"] == previous["frame_id"] + 1
    )

    increasing_time = (
        current["timestamp"] > previous["timestamp"]
    )

    valid_positions = (
        previous["status"] in VALID_STATUSES
        and current["status"] in VALID_STATUSES
    )

    return (
        same_stream
        and consecutive_frames
        and increasing_time
        and valid_positions
    )


def calculate_kinematics(events):
    """
    Calculate movement metrics from processed ball positions.

    Missing or discontinuous intervals are not included.
    The input dataset is never modified.
    """

    result = deepcopy(events)

    total_distance = 0.0
    valid_duration = 0.0
    interval_count = 0

    speeds = []
    accelerations = []

    previous = None
    previous_velocity = None

    for event in result:

        # Default values for events without a valid interval
        event["kinematics"] = {
            "distance_m": None,
            "speed_m_s": None,
            "velocity_x_m_s": None,
            "velocity_y_m_s": None,
            "acceleration_m_s2": None,
            "contains_prediction": None
        }

        if previous is not None and can_connect(
            previous, event
        ):
            dt = event["timestamp"] - previous["timestamp"]

            x1 = previous["position"]["x"]
            y1 = previous["position"]["y"]

            x2 = event["position"]["x"]
            y2 = event["position"]["y"]

            dx = x2 - x1
            dy = y2 - y1

            distance = math.hypot(dx, dy)

            vx = dx / dt
            vy = dy / dt

            speed = distance / dt

            event["kinematics"].update({
                "distance_m": round(distance, 3),
                "speed_m_s": round(speed, 3),
                "velocity_x_m_s": round(vx, 3),
                "velocity_y_m_s": round(vy, 3),
                "contains_prediction": (
                    previous["status"] == "predicted"
                    or event["status"] == "predicted"
                )
            })

            total_distance += distance
            valid_duration += dt
            interval_count += 1
            speeds.append(speed)

            # Velocity corresponds to the interval midpoint
            midpoint = (
                previous["timestamp"] + event["timestamp"]
            ) / 2

            # Calculate acceleration using consecutive
            # velocity estimates
            if previous_velocity is not None:

                previous_vx, previous_vy, previous_midpoint = (
                    previous_velocity
                )

                velocity_dt = midpoint - previous_midpoint

                if velocity_dt > 0:
                    ax = (vx - previous_vx) / velocity_dt
                    ay = (vy - previous_vy) / velocity_dt

                    acceleration = math.hypot(ax, ay)

                    event["kinematics"]["acceleration_m_s2"] = (
                        round(acceleration, 3)
                    )

                    accelerations.append(acceleration)

            previous_velocity = (vx, vy, midpoint)

        else:
            # Reset when an interval is broken
            previous_velocity = None

        previous = event

    observed_count = sum(
        event["status"] == "observed" for event in result
    )

    predicted_count = sum(
        event["status"] == "predicted" for event in result
    )

    missing_count = sum(
        event["status"] == "missing" for event in result
    )

    summary = {
        "total_events": len(result),
        "observed_events": observed_count,
        "predicted_events": predicted_count,
        "missing_events": missing_count,
        "valid_intervals": interval_count,
        "valid_duration_s": round(valid_duration, 3),
        "total_distance_m": round(total_distance, 3),
        "average_speed_m_s": (
            round(total_distance / valid_duration, 3)
            if valid_duration > 0 else None
        ),
        "maximum_speed_m_s": (
            round(max(speeds), 3) if speeds else None
        ),
        "maximum_acceleration_m_s2": (
            round(max(accelerations), 3)
            if accelerations else None
        )
    }

    return result, summary


def main():
    if not INPUT.exists():
        raise FileNotFoundError(f"Input not found: {INPUT}")

    with INPUT.open("r", encoding="utf-8") as file:
        events = [
            json.loads(line)
            for line in file
            if line.strip()
        ]

    enriched, summary = calculate_kinematics(events)

    ENRICHED_OUTPUT.parent.mkdir(
        parents=True, exist_ok=True
    )

    with ENRICHED_OUTPUT.open(
        "w", encoding="utf-8"
    ) as file:
        for event in enriched:
            file.write(json.dumps(event) + "\n")

    with SUMMARY_OUTPUT.open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(summary, file, indent=2)

    print("Ball movement analytics completed")
    print(json.dumps(summary, indent=2))
    print(f"Enriched data: {ENRICHED_OUTPUT}")
    print(f"Summary: {SUMMARY_OUTPUT}")


if __name__ == "__main__":
    main()
