
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

GROUND_TRUTH = ROOT / "data/raw/ball_tracking_match_001.jsonl"

DATASETS = {
    "noisy": (
        ROOT / "data/raw/ball_tracking_match_001_noisy.jsonl"
    ),
    "reconstructed": (
        ROOT / "data/processed/"
        "ball_tracking_match_001_reconstructed.jsonl"
    ),
    "smoothed": (
        ROOT / "data/processed/"
        "ball_tracking_match_001_smoothed.jsonl"
    ),
}

OUTPUT = (
    ROOT / "data/processed/"
    "ball_tracking_match_001_accuracy_report.json"
)


def read_jsonl(path):
    with path.open("r", encoding="utf-8") as file:
        return [
            json.loads(line)
            for line in file
            if line.strip()
        ]


def index_events(events):
    """Index events by match, ball and frame."""
    indexed = {}

    for event in events:
        key = (
            event["match_id"],
            event["ball_id"],
            event["frame_id"]
        )

        if key in indexed:
            raise ValueError(f"Duplicate tracking key: {key}")

        indexed[key] = event

    return indexed


def position_errors(ground_truth, estimates):
    """
    Compare estimated positions with ground truth.

    Returns frame-keyed pairs of (status, error).
    Missing positions have error=None.
    """

    truth = index_events(ground_truth)
    measured = index_events(estimates)

    if not truth or truth.keys() != measured.keys():
        raise ValueError(
            "Reference and estimate frame keys must match"
        )

    errors = {}

    for key, actual in truth.items():
        event = measured[key]

        if abs(
            actual["timestamp"] - event["timestamp"]
        ) > 1e-6:
            raise ValueError(f"Timestamp mismatch at {key}")

        for field in ("coordinate_frame", "position_unit"):
            if actual[field] != event[field]:
                raise ValueError(f"{field} mismatch at {key}")

        status = event["status"]
        position = event["position"]

        if status == "missing":
            if (
                position["x"] is not None
                or position["y"] is not None
            ):
                raise ValueError(
                    f"Missing event contains position at {key}"
                )

            errors[key] = (status, None)

        elif status in ("observed", "predicted"):
            coordinates = (
                actual["position"]["x"],
                actual["position"]["y"],
                position["x"],
                position["y"]
            )

            if any(
                isinstance(v, bool)
                or not isinstance(v, (int, float))
                or not math.isfinite(v)
                for v in coordinates
            ):
                raise ValueError(f"Invalid position at {key}")

            error = math.hypot(
                position["x"] - actual["position"]["x"],
                position["y"] - actual["position"]["y"]
            )

            errors[key] = (status, error)

        else:
            raise ValueError(
                f"Unknown status at {key}: {status}"
            )

    return errors


def percentile(values, fraction):
    """Calculate an interpolated percentile."""

    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction

    lower = math.floor(index)
    upper = math.ceil(index)

    return (
        ordered[lower]
        + (index - lower)
        * (ordered[upper] - ordered[lower])
    )


def summarize_errors(errors):
    """Calculate error metrics and distribution."""

    if not errors:
        return {
            "count": 0,
            "mae_m": None,
            "rmse_m": None,
            "median_m": None,
            "p95_m": None,
            "max_m": None,
            "histogram_m": {}
        }

    bands = [
        ("<0.10", 0, 0.10),
        ("0.10-0.25", 0.10, 0.25),
        ("0.25-0.50", 0.25, 0.50),
        ("0.50-1.00", 0.50, 1.00),
        (">=1.00", 1.00, math.inf)
    ]

    n = len(errors)

    return {
        "count": n,
        "mae_m": round(sum(errors) / n, 4),
        "rmse_m": round(
            math.sqrt(sum(e * e for e in errors) / n), 4
        ),
        "median_m": round(percentile(errors, 0.5), 4),
        "p95_m": round(percentile(errors, 0.95), 4),
        "max_m": round(max(errors), 4),
        "histogram_m": {
            name: sum(low <= e < high for e in errors)
            for name, low, high in bands
        }
    }


def evaluate_dataset(ground_truth, estimates):
    aligned = position_errors(ground_truth, estimates)

    available = [
        error
        for _, error in aligned.values()
        if error is not None
    ]

    return {
        "total_frames": len(aligned),
        "available_frames": len(available),
        "missing_frames": len(aligned) - len(available),
        "coverage_pct": round(
            100 * len(available) / len(aligned), 2
        ),
        "all_available": summarize_errors(available),
        "observed_only": summarize_errors([
            error
            for status, error in aligned.values()
            if status == "observed"
        ]),
        "predicted_only": summarize_errors([
            error
            for status, error in aligned.values()
            if status == "predicted"
        ])
    }


def paired_comparison(ground_truth, baseline, candidate):
    """Compare two methods on the same available frames."""

    before = position_errors(ground_truth, baseline)
    after = position_errors(ground_truth, candidate)

    common = [
        key for key in before
        if before[key][1] is not None
        and after[key][1] is not None
    ]

    baseline_errors = [
        before[key][1] for key in common
    ]

    candidate_errors = [
        after[key][1] for key in common
    ]

    a = summarize_errors(baseline_errors)
    b = summarize_errors(candidate_errors)

    if baseline_errors and sum(baseline_errors) > 0:
        reduction = round(
            100 * (
                1 - sum(candidate_errors)
                / sum(baseline_errors)
            ), 2
        )
    else:
        reduction = None

    return {
        "common_frames": len(common),
        "baseline_mae_m": a["mae_m"],
        "candidate_mae_m": b["mae_m"],
        "baseline_rmse_m": a["rmse_m"],
        "candidate_rmse_m": b["rmse_m"],
        "mae_reduction_pct": reduction
    }


def main():
    truth = read_jsonl(GROUND_TRUTH)

    datasets = {
        name: read_jsonl(path)
        for name, path in DATASETS.items()
    }

    report = {
        "metric_definition": (
            "2D Euclidean position error (meters)"
        ),
        "datasets": {
            name: evaluate_dataset(truth, events)
            for name, events in datasets.items()
        },
        "paired_comparisons": {
            "noisy_to_smoothed": paired_comparison(
                truth,
                datasets["noisy"],
                datasets["smoothed"]
            ),
            "reconstructed_to_smoothed": paired_comparison(
                truth,
                datasets["reconstructed"],
                datasets["smoothed"]
            )
        }
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    OUTPUT.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8"
    )

    for name, details in report["datasets"].items():
        metrics = details["all_available"]

        print(
            f"{name:14} "
            f"coverage={details['coverage_pct']:6.2f}% "
            f"MAE={metrics['mae_m']}m "
            f"RMSE={metrics['rmse_m']}m"
        )

    print("\nPaired comparisons:")
    print(json.dumps(report["paired_comparisons"], indent=2))

    print(f"\nReport saved to: {OUTPUT}")


if __name__ == "__main__":
    main()
