
import json
import os
from datetime import datetime
from pathlib import Path

import psycopg
from dotenv import load_dotenv
from psycopg.types.json import Jsonb

from src.validate import SCHEMA, validate_event
from src.quality import check_sequence

ROOT = Path(__file__).resolve().parents[1]

load_dotenv(ROOT / ".env")

DATASETS = {
    "noisy": ROOT / "data/raw/ball_tracking_match_001_noisy.jsonl",
    "reconstructed": ROOT / "data/processed/ball_tracking_match_001_reconstructed.jsonl",
    "smoothed": ROOT / "data/processed/ball_tracking_match_001_smoothed.jsonl",
    "enriched": ROOT / "data/processed/ball_tracking_match_001_enriched.jsonl"
}

METRICS = (
    ROOT / "data/processed/ball_tracking_match_001_metrics.json"
)

ACCURACY = (
    ROOT / "data/processed/ball_tracking_match_001_accuracy_report.json"
)


def read_jsonl(path):
    with path.open("r", encoding="utf-8") as file:
        return [
            json.loads(line)
            for line in file
            if line.strip()
        ]


def event_to_row(event, stage):
    if stage not in DATASETS:
        raise ValueError(f"Unknown processing stage: {stage}")

    position = event["position"]

    return (
        event["match_id"],
        event["ball_id"],
        stage,
        event["frame_id"],
        event["source"],
        event["timestamp"],
        datetime.fromisoformat(
            event["captured_at_utc"].replace("Z", "+00:00")
        ),
        event["coordinate_frame"],
        event["position_unit"],
        position["x"],
        position["y"],
        event["status"],
        event["confidence"],
        event["last_observed_time"],
        Jsonb(event["uncertainty"])
            if event["uncertainty"] is not None else None,
        Jsonb(event["processing"])
            if event.get("processing") is not None else None,
        Jsonb(event["kinematics"])
            if event.get("kinematics") is not None else None
    )


INSERT_EVENT = """
INSERT INTO tracking_events (
    match_id, ball_id, stage, frame_id,
    source, event_time_s, captured_at,
    coordinate_frame, position_unit,
    x_m, y_m, status, confidence, last_observed_s,
    uncertainty, processing, kinematics
)
VALUES (
    %s, %s, %s, %s,
    %s, %s, %s,
    %s, %s,
    %s, %s, %s, %s, %s,
    %s, %s, %s
)
ON CONFLICT (match_id, ball_id, stage, frame_id)
DO UPDATE SET
    source = EXCLUDED.source,
    event_time_s = EXCLUDED.event_time_s,
    captured_at = EXCLUDED.captured_at,
    coordinate_frame = EXCLUDED.coordinate_frame,
    position_unit = EXCLUDED.position_unit,
    x_m = EXCLUDED.x_m,
    y_m = EXCLUDED.y_m,
    status = EXCLUDED.status,
    confidence = EXCLUDED.confidence,
    last_observed_s = EXCLUDED.last_observed_s,
    uncertainty = EXCLUDED.uncertainty,
    processing = EXCLUDED.processing,
    kinematics = EXCLUDED.kinematics
"""


def main():
    password = os.getenv("POSTGRES_PASSWORD")

    if not password:
        raise RuntimeError("POSTGRES_PASSWORD is not configured")

    # Validate every dataset before opening a transaction.
    prepared = {}

    for stage, path in DATASETS.items():
        events = read_jsonl(path)

        for i, event in enumerate(events, start=1):
            errors = validate_event(event, SCHEMA)
            if errors:
                raise ValueError(
                    f"{stage}, record {i}: {errors}"
                )

        quality = check_sequence(
            events, SCHEMA["sampling_rate_hz"]
        )

        if not quality["passed"]:
            raise ValueError(f"Sequence check failed: {stage}")

        prepared[stage] = events

    identities = {
        (e["match_id"], e["ball_id"])
        for events in prepared.values()
        for e in events
    }

    if len(identities) != 1:
        raise ValueError(
            "Step 11 expects one match and ball per batch"
        )

    match_id, ball_id = identities.pop()

    metrics = json.loads(METRICS.read_text(encoding="utf-8"))
    accuracy = json.loads(ACCURACY.read_text(encoding="utf-8"))

    with psycopg.connect(
        host="127.0.0.1",
        port=int(os.getenv("POSTGRES_PORT", "5433")),
        dbname=os.getenv("POSTGRES_DB"),
        user=os.getenv("POSTGRES_USER"),
        password=password
    ) as conn:

        with conn.cursor() as cur:

            for stage, events in prepared.items():
                rows = [
                    event_to_row(event, stage)
                    for event in events
                ]

                cur.executemany(INSERT_EVENT, rows)

                print(
                    f"Loaded {len(rows)} records: {stage}"
                )

            cur.execute("""
                INSERT INTO match_metrics (
                    match_id, ball_id, stage, metrics
                )
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (match_id, ball_id, stage)
                DO UPDATE SET
                    metrics = EXCLUDED.metrics,
                    updated_at = NOW()
            """, (
                match_id,
                ball_id,
                "enriched",
                Jsonb(metrics)
            ))

            cur.execute("""
                INSERT INTO accuracy_reports (
                    match_id, ball_id, report
                )
                VALUES (%s, %s, %s)
                ON CONFLICT (match_id, ball_id)
                DO UPDATE SET
                    report = EXCLUDED.report,
                    updated_at = NOW()
            """, (
                match_id,
                ball_id,
                Jsonb(accuracy)
            ))

        # Successful exit commits the entire transaction.

    print("Database loading completed successfully")


if __name__ == "__main__":
    main()
