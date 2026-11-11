
import json
from copy import deepcopy
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

INPUT = (
    ROOT / "data/raw/ball_tracking_match_001_noisy.jsonl"
)

OUTPUT = (
    ROOT / "data/processed/"
    "ball_tracking_match_001_reconstructed.jsonl"
)

# Maximum missing interval we will reconstruct.
# At 10 Hz, 25 frames represent 2.5 seconds.
MAX_GAP_FRAMES = 25


def reconstruct_trajectory(events, max_gap_frames=25):
    """
    Reconstruct short missing intervals using linear
    interpolation between two observed positions.

    Does not use ground-truth positions.
    Does not extrapolate at the beginning or end.
    """

    if max_gap_frames < 0:
        raise ValueError("max_gap_frames must be non-negative")

    result = deepcopy(events)
    filled_count = 0
    i = 0

    while i < len(result):

        if result[i]["status"] != "missing":
            i += 1
            continue

        # Find the full consecutive missing interval
        gap_start = i

        while (
            i < len(result)
            and result[i]["status"] == "missing"
        ):
            i += 1

        gap_end = i
        gap_size = gap_end - gap_start

        # Need observations on both sides
        if gap_start == 0 or gap_end == len(result):
            continue

        # Avoid reconstructing excessively long gaps
        if gap_size > max_gap_frames:
            continue

        before = result[gap_start - 1]
        after = result[gap_end]

        if (
            before["status"] != "observed"
            or after["status"] != "observed"
        ):
            continue

        t0 = before["timestamp"]
        t1 = after["timestamp"]

        if t1 <= t0:
            continue

        x0 = before["position"]["x"]
        y0 = before["position"]["y"]

        x1 = after["position"]["x"]
        y1 = after["position"]["y"]

        # Interpolate every missing record in the interval
        for j in range(gap_start, gap_end):

            event = result[j]
            t = event["timestamp"]

            if not t0 < t < t1:
                continue

            alpha = (t - t0) / (t1 - t0)

            x = x0 + alpha * (x1 - x0)
            y = y0 + alpha * (y1 - y0)

            event["position"] = {
                "x": round(x, 3),
                "y": round(y, 3)
            }

            event["status"] = "predicted"
            event["source"] = "trajectory_interpolator_v1"

            # Preserve the last real observation time
            event["last_observed_time"] = t0

            # No calibrated prediction confidence yet
            event["confidence"] = None
            event["uncertainty"] = None

            filled_count += 1

    return result, filled_count


def main():
    if not INPUT.exists():
        raise FileNotFoundError(f"Input not found: {INPUT}")

    with INPUT.open("r", encoding="utf-8") as file:
        events = [
            json.loads(line)
            for line in file
            if line.strip()
        ]

    reconstructed, filled = reconstruct_trajectory(
        events,
        MAX_GAP_FRAMES
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT.open("w", encoding="utf-8") as file:
        for event in reconstructed:
            file.write(json.dumps(event) + "\n")

    counts = Counter(
        event["status"] for event in reconstructed
    )

    print("Trajectory reconstruction completed")
    print(f"Total events: {len(reconstructed)}")
    print(f"Observed: {counts['observed']}")
    print(f"Predicted: {counts['predicted']}")
    print(f"Still missing: {counts['missing']}")
    print(f"Positions reconstructed: {filled}")
    print(f"Output: {OUTPUT}")


if __name__ == "__main__":
    main()
