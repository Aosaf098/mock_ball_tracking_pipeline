
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

GROUND_TRUTH = (
    ROOT / "data/raw/ball_tracking_match_001.jsonl"
)

OUTPUT = (
    ROOT / "data/raw/ball_tracking_match_001_noisy.jsonl"
)

# Reproducible random simulation
rng = random.Random(42)

# Camera simulation parameters
NOISE_STD_M = 0.15
RANDOM_MISS_PROBABILITY = 0.05

# Occlusion windows, specified by frame IDs
OCCLUSIONS = [
    (70, 90),
    (180, 195)
]


def is_occluded(frame_id):
    return any(
        start <= frame_id < end
        for start, end in OCCLUSIONS
    )


def main():
    if not GROUND_TRUTH.exists():
        raise FileNotFoundError(
            f"Ground-truth file not found: {GROUND_TRUTH}"
        )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    last_observed_time = None
    counts = {"observed": 0, "missing": 0}

    with (
        GROUND_TRUTH.open("r", encoding="utf-8") as source,
        OUTPUT.open("w", encoding="utf-8") as target
    ):
        for line in source:
            if not line.strip():
                continue

            event = json.loads(line)

            frame_id = event["frame_id"]
            timestamp = event["timestamp"]

            # Model occlusion and random detection failure
            missing = (
                is_occluded(frame_id)
                or rng.random() < RANDOM_MISS_PROBABILITY
            )

            if missing:
                event["position"] = {
                    "x": None,
                    "y": None
                }

                event["status"] = "missing"
                event["confidence"] = None
                event["uncertainty"] = None

            else:
                true_x = event["position"]["x"]
                true_y = event["position"]["y"]

                # Add Gaussian measurement noise
                measured_x = true_x + rng.gauss(
                    0, NOISE_STD_M
                )
                measured_y = true_y + rng.gauss(
                    0, NOISE_STD_M
                )

                event["position"] = {
                    "x": round(measured_x, 3),
                    "y": round(measured_y, 3)
                }

                event["status"] = "observed"
                event["confidence"] = None

                event["uncertainty"] = {
                    "sigma_x_m": NOISE_STD_M,
                    "sigma_y_m": NOISE_STD_M
                }

                last_observed_time = timestamp

            event["source"] = "simulated_camera_01"
            event["last_observed_time"] = last_observed_time

            counts[event["status"]] += 1

            target.write(json.dumps(event) + "\n")

    print("Camera simulation completed")
    print(f"Observed events: {counts['observed']}")
    print(f"Missing events: {counts['missing']}")
    print(f"Total events: {sum(counts.values())}")
    print(f"Output: {OUTPUT}")


if __name__ == "__main__":
    main()
