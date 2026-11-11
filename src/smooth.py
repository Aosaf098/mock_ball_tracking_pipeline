
import json
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

INPUT = (
    ROOT / "data/processed/"
    "ball_tracking_match_001_reconstructed.jsonl"
)

OUTPUT = (
    ROOT / "data/processed/"
    "ball_tracking_match_001_smoothed.jsonl"
)

WINDOW_SIZE = 5
POSITION_STATUSES = {"observed", "predicted"}


def same_stream(previous, current):
    """Prevent smoothing across different tracking streams."""

    fields = (
        "match_id",
        "ball_id",
        "coordinate_frame",
        "position_unit"
    )

    same_identity = all(
        previous.get(field) == current.get(field)
        for field in fields
    )

    consecutive_frames = (
        current["frame_id"] == previous["frame_id"] + 1
    )

    increasing_time = (
        current["timestamp"] > previous["timestamp"]
    )

    return (
        same_identity
        and consecutive_frames
        and increasing_time
    )


def smooth_trajectory(events, window_size=5):
    """
    Smooth valid tracking positions with a centered
    triangular moving-average filter.

    Missing positions remain unchanged.
    No smoothing across missing frames or stream boundaries.
    The input events are never modified.
    """

    if window_size < 1 or window_size % 2 == 0:
        raise ValueError(
            "window_size must be a positive odd integer"
        )

    result = deepcopy(events)
    radius = window_size // 2
    filtered_count = 0

    i = 0

    while i < len(events):

        # Skip missing tracking events
        if events[i]["status"] not in POSITION_STATUSES:
            i += 1
            continue

        # Identify a continuous segment of valid positions
        start = i
        end = i + 1

        while end < len(events):
            current = events[end]
            previous = events[end - 1]

            if current["status"] not in POSITION_STATUSES:
                break

            if not same_stream(previous, current):
                break

            end += 1

        # Smooth only points with a complete window
        # Leave segment boundaries unchanged
        for j in range(start + radius, end - radius):

            weighted_x = 0.0
            weighted_y = 0.0
            total_weight = 0

            for k in range(j - radius, j + radius + 1):

                weight = radius + 1 - abs(k - j)
                position = events[k]["position"]

                weighted_x += position["x"] * weight
                weighted_y += position["y"] * weight
                total_weight += weight

            result[j]["position"] = {
                "x": round(weighted_x / total_weight, 3),
                "y": round(weighted_y / total_weight, 3)
            }

            # Raw measurement uncertainty does not
            # directly describe the filtered estimate.
            result[j]["uncertainty"] = None

            result[j]["processing"] = {
                "method": "triangular_moving_average",
                "window_size": window_size
            }

            filtered_count += 1

        i = end

    return result, filtered_count


def main():
    if not INPUT.exists():
        raise FileNotFoundError(f"Input not found: {INPUT}")

    with INPUT.open("r", encoding="utf-8") as file:
        events = [
            json.loads(line)
            for line in file
            if line.strip()
        ]

    smoothed, count = smooth_trajectory(
        events,
        WINDOW_SIZE
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT.open("w", encoding="utf-8") as file:
        for event in smoothed:
            file.write(json.dumps(event) + "\n")

    print("Trajectory smoothing completed")
    print(f"Total events: {len(smoothed)}")
    print(f"Filtered positions: {count}")
    print(f"Output: {OUTPUT}")


if __name__ == "__main__":
    main()
