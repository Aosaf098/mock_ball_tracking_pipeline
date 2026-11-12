
-- Mock Ball Tracking Database Schema v1

CREATE TABLE IF NOT EXISTS tracking_events (
    match_id TEXT NOT NULL,
    ball_id TEXT NOT NULL,
    stage TEXT NOT NULL,
    frame_id INTEGER NOT NULL,

    source TEXT NOT NULL,
    event_time_s DOUBLE PRECISION NOT NULL,
    captured_at TIMESTAMPTZ NOT NULL,

    coordinate_frame TEXT NOT NULL,
    position_unit TEXT NOT NULL,

    x_m DOUBLE PRECISION,
    y_m DOUBLE PRECISION,

    status TEXT NOT NULL,
    confidence DOUBLE PRECISION,
    last_observed_s DOUBLE PRECISION,

    uncertainty JSONB,
    processing JSONB,
    kinematics JSONB,

    ingested_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    PRIMARY KEY (match_id, ball_id, stage, frame_id),

    CONSTRAINT valid_stage CHECK (
        stage IN (
            'noisy',
            'reconstructed',
            'smoothed',
            'enriched'
        )
    ),

    CONSTRAINT valid_status CHECK (
        status IN ('observed', 'predicted', 'missing')
    ),

    CONSTRAINT valid_position CHECK (
        (status = 'missing'
            AND x_m IS NULL AND y_m IS NULL)
        OR
        (status IN ('observed', 'predicted')
            AND x_m IS NOT NULL AND y_m IS NOT NULL)
    ),

    CONSTRAINT valid_confidence CHECK (
        confidence IS NULL
        OR confidence BETWEEN 0 AND 1
    ),

    CONSTRAINT valid_time CHECK (
        event_time_s >= 0
        AND frame_id >= 0
        AND (
            last_observed_s IS NULL
            OR last_observed_s BETWEEN 0 AND event_time_s
        )
    )
);

CREATE INDEX IF NOT EXISTS idx_tracking_time
ON tracking_events (
    match_id,
    ball_id,
    stage,
    event_time_s
);

-- Aggregate movement statistics
CREATE TABLE IF NOT EXISTS match_metrics (
    match_id TEXT NOT NULL,
    ball_id TEXT NOT NULL,
    stage TEXT NOT NULL,
    metrics JSONB NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    PRIMARY KEY (match_id, ball_id, stage)
);

-- Simulation accuracy evaluation reports
CREATE TABLE IF NOT EXISTS accuracy_reports (
    match_id TEXT NOT NULL,
    ball_id TEXT NOT NULL,
    report JSONB NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    PRIMARY KEY (match_id, ball_id)
);
