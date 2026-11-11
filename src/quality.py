
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

INPUT = ROOT / "data/processed/ball_tracking_match_001_validated.jsonl"
OUTPUT = ROOT / "data/processed/quality_report.json"

SCHEMA_PATH = ROOT / "config/schema.json"


def check_sequence(events, sampling_rate_hz, tolerance=0.005):
    """Check sequence-level quality of tracking events."""

    if sampling_rate_hz <= 0:
        raise ValueError("Sampling rate must be positive")

    expected_interval = 1.0 / sampling_rate_hz

    seen_frames = set()
    duplicate_frames = []
    out_of_order_lines = []
    non_increasing_timestamp_lines = []
    timing_mismatch_lines = []

    previous = None

    for line_number, event in enumerate(events, start=1):
        frame_id = event["frame_id"]
        timestamp = event["timestamp"]

        # Detect duplicate frame IDs
        if frame_id in seen_frames:
            duplicate_frames.append(frame_id)

        seen_frames.add(frame_id)

        if previous is not None:
            frame_difference = frame_id - previous["frame_id"]
            time_difference = timestamp - previous["timestamp"]

            # Detect out-of-order frames
            if frame_difference < 0:
                out_of_order_lines.append(line_number)

            # Detect non-increasing timestamps
            if time_difference <= 0:
                non_increasing_timestamp_lines.append(line_number)

            # Check timing against the sampling rate
            if frame_difference > 0:
                expected_time = frame_difference * expected_interval

                if abs(time_difference - expected_time) > tolerance:
                    timing_mismatch_lines.append(line_number)

        previous = event

    # Detect missing IDs within the observed frame range
    if seen_frames:
        expected_frames = set(
            range(min(seen_frames), max(seen_frames) + 1)
        )
        missing_frames = sorted(expected_frames - seen_frames)
    else:
        missing_frames = []

    report = {
        "total_records": len(events),
        "duplicate_frames": duplicate_frames,
        "missing_frames": missing_frames,
        "out_of_order_lines": out_of_order_lines,
        "non_increasing_timestamp_lines": non_increasing_timestamp_lines,
        "timing_mismatch_lines": timing_mismatch_lines
    }

    report["passed"] = not any(
        report[key] for key in (
            "duplicate_frames",
            "missing_frames",
            "out_of_order_lines",
            "non_increasing_timestamp_lines",
            "timing_mismatch_lines"
        )
    )

    return report


def main():
    with SCHEMA_PATH.open("r", encoding="utf-8") as file:
        schema = json.load(file)

    with INPUT.open("r", encoding="utf-8") as file:
        events = [
            json.loads(line)
            for line in file
            if line.strip()
        ]

    report = check_sequence(
        events,
        schema["sampling_rate_hz"]
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
