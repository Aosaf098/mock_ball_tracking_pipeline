
import json
import math
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

RAW = ROOT / "data/raw/ball_tracking_match_001.jsonl"
VALID = ROOT / "data/processed/ball_tracking_match_001_validated.jsonl"
REJECTED = ROOT / "data/rejected/ball_tracking_match_001_rejected.jsonl"

with open(ROOT / "config/schema.json", encoding="utf-8") as f:
    SCHEMA = json.load(f)


def is_number(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def validate_event(event, schema):
    errors = []

    if not isinstance(event, dict):
        return ["Event must be a JSON object"]

    # 1. Required fields
    for field in schema["required_fields"]:
        if field not in event:
            errors.append(f"Missing field: {field}")

    # 2. Identifiers
    for field in ("match_id", "ball_id", "source"):
        value = event.get(field)
        if not isinstance(value, str) or not value.strip():
            errors.append(f"Invalid {field}")

    # 3. Timestamp
    timestamp = event.get("timestamp")

    if not is_number(timestamp) or timestamp < 0:
        errors.append("Invalid timestamp")

    # 4. UTC capture time
    captured_at = event.get("captured_at_utc")

    try:
        if not isinstance(captured_at, str) or not captured_at.endswith("Z"):
            raise ValueError("Expected UTC timestamp ending in Z")

        datetime.fromisoformat(captured_at.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        errors.append("Invalid captured_at_utc")

    # 5. Frame ID
    frame_id = event.get("frame_id")

    if (
        not isinstance(frame_id, int)
        or isinstance(frame_id, bool)
        or frame_id < 0
    ):
        errors.append("Invalid frame_id")

    # 6. Coordinate system
    coordinate_system = schema["coordinate_system"]

    if event.get("coordinate_frame") != coordinate_system["frame"]:
        errors.append("Coordinate frame mismatch")

    if event.get("position_unit") != coordinate_system["unit"]:
        errors.append("Position unit mismatch")

    # 7. Tracking status
    status = event.get("status")

    if status not in schema["allowed_status"]:
        errors.append("Invalid tracking status")

    # 8. Position
    position = event.get("position")

    if not isinstance(position, dict):
        errors.append("Position must be an object")
    else:
        x = position.get("x")
        y = position.get("y")

        if status in schema["validation_rules"]["position_required_for"]:
            if not is_number(x) or not is_number(y):
                errors.append("Position required for observed/predicted")

        if status in schema["validation_rules"]["position_null_for"]:
            if x is not None or y is not None:
                errors.append("Missing status requires null coordinates")

    # 9. Confidence
    confidence = event.get("confidence")

    if confidence is not None:
        limits = schema["confidence_range"]

        if (
            not is_number(confidence)
            or not limits["min"] <= confidence <= limits["max"]
        ):
            errors.append("Confidence must be between 0 and 1")

    # 10. Last observed time
    last_time = event.get("last_observed_time")

    if last_time is not None:
        if not is_number(last_time) or last_time < 0:
            errors.append("Invalid last_observed_time")
        elif is_number(timestamp) and last_time > timestamp:
            errors.append("Last observed time exceeds timestamp")

    # 11. Uncertainty
    uncertainty = event.get("uncertainty")

    if uncertainty is not None:
        if not isinstance(uncertainty, dict):
            errors.append("Uncertainty must be an object or null")
        else:
            for field in schema["uncertainty_fields"]:
                value = uncertainty.get(field)

                if not is_number(value) or value < 0:
                    errors.append(f"Invalid uncertainty field: {field}")

    return errors


def main():
    VALID.parent.mkdir(parents=True, exist_ok=True)
    REJECTED.parent.mkdir(parents=True, exist_ok=True)

    valid_count = 0
    rejected_count = 0

    with (
        RAW.open("r", encoding="utf-8") as source,
        VALID.open("w", encoding="utf-8") as valid_file,
        REJECTED.open("w", encoding="utf-8") as rejected_file
    ):
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue

            try:
                event = json.loads(line)
                errors = validate_event(event, SCHEMA)
            except json.JSONDecodeError:
                event = None
                errors = ["Invalid JSON syntax"]

            if errors:
                rejected_count += 1

                rejected_file.write(json.dumps({
                    "line_number": line_number,
                    "errors": errors,
                    "raw_record": line.strip()
                }) + "\n")
            else:
                valid_count += 1
                valid_file.write(json.dumps(event) + "\n")

    print("Validation completed")
    print(f"Valid records: {valid_count}")
    print(f"Rejected records: {rejected_count}")
    print(f"Validated file: {VALID}")
    print(f"Rejected file: {REJECTED}")


if __name__ == "__main__":
    main()
