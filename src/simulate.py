
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Simulation settings
MATCH_ID = "match_001"
BALL_ID = "ball_01"

SAMPLING_RATE = 10
DURATION = 30
DT = 1.0 / SAMPLING_RATE

FIELD_LENGTH = 105.0
FIELD_WIDTH = 68.0

# Coordinate origin is at the pitch center
X_LIMIT = FIELD_LENGTH / 2
Y_LIMIT = FIELD_WIDTH / 2

# Initial ball position and velocity
x, y = 0.0, 0.0
vx, vy = 4.0, 2.0

# Output location
ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data" / "raw" / "ball_tracking_match_001.jsonl"

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

start_time = datetime.now(timezone.utc)

with OUTPUT.open("w", encoding="utf-8") as file:

    for frame_id in range(SAMPLING_RATE * DURATION):

        timestamp = round(frame_id * DT, 3)

        captured_at = start_time + timedelta(
            seconds=timestamp
        )

        event = {
            "match_id": MATCH_ID,
            "ball_id": BALL_ID,
            "source": "simulator",
            "timestamp": timestamp,
            "captured_at_utc": captured_at.isoformat(
                timespec="milliseconds"
            ).replace("+00:00", "Z"),
            "frame_id": frame_id,
            "position": {
                "x": round(x, 3),
                "y": round(y, 3)
            },
            "coordinate_frame": "pitch_center",
            "position_unit": "meters",
            "status": "observed",
            "confidence": None,
            "last_observed_time": timestamp,
            "uncertainty": None
        }

        file.write(json.dumps(event) + "\n")

        # Update the ball position
        x += vx * DT
        y += vy * DT

        # Reflect the ball at pitch boundaries
        if x > X_LIMIT:
            x = 2 * X_LIMIT - x
            vx = -vx
        elif x < -X_LIMIT:
            x = -2 * X_LIMIT - x
            vx = -vx

        if y > Y_LIMIT:
            y = 2 * Y_LIMIT - y
            vy = -vy
        elif y < -Y_LIMIT:
            y = -2 * Y_LIMIT - y
            vy = -vy

print(f"Generated {SAMPLING_RATE * DURATION} events")
print(f"Saved to: {OUTPUT}")
