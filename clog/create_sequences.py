import argparse
import pickle
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
RESOURCES_ROOT = BASE_DIR / "data" / "NOVA" / "resources"

BASE_COLUMNS = [
    "Content",
    "level",
    "service",
    "round_1",
    "round_2",
    "api_round_1",
    "api_round_2",
    "assertions_round_1",
    "assertions_round_2",
    "clusters",
    "round",
    "anom_label",
    "encoded_labels",
    "time_hour_day",
]


def to_list(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    return [] if value in (None, "") else [value]


def format_numeric_sequence(seq: Sequence[int], joiner: str = " ") -> str:
    if not seq:
        return ""
    return joiner.join(str(int(x)) for x in seq)


def format_list(seq: Sequence[object]) -> str:
    if not seq:
        return "[]"
    return "[" + ", ".join(str(x) for x in seq) + "]"


def derive_records(data: dict, interval: str) -> pd.DataFrame:
    records: List[dict] = []
    for name, windows in data.items():
        for window in windows:
            base = {column: to_list(value) for column, value in zip(BASE_COLUMNS, window)}
            parts = name.split("_", 3)
            test_id = parts[0] if len(parts) > 0 else ""
            parent_service = parts[1] if len(parts) > 1 else ""
            round_id = parts[2] if len(parts) > 2 else ""

            clusters = [int(x) for x in base["clusters"]]
            encoded_labels = [int(x) for x in base["encoded_labels"]]
            anomaly_flags = [int(x) for x in base["anom_label"]]
            round_1 = [str(x) for x in base["round_1"]]
            round_2 = [str(x) for x in base["round_2"]]
            api_round_1 = [str(x) for x in base["api_round_1"]]
            api_round_2 = [str(x) for x in base["api_round_2"]]
            assertions_round_1 = [str(x) for x in base["assertions_round_1"]]
            assertions_round_2 = [str(x) for x in base["assertions_round_2"]]
            time_points = [str(x) for x in base["time_hour_day"]]
            services = [str(x) for x in base["service"]]
            levels = [str(x) for x in base["level"]]
            contents = [str(x) for x in base["Content"]]

            anomaly_count = int(np.sum(anomaly_flags)) if anomaly_flags else 0
            final_status = "FAILURE" if ("FAILURE" in round_1 or anomaly_count > 0) else "NO_FAILURE"

            records.append(
                {
                    "interval": interval,
                    "experiment_id": name,
                    "test_id": test_id,
                    "parent_service": parent_service,
                    "round": round_id,
                    "window_start": time_points[0] if time_points else "",
                    "window_end": time_points[-1] if time_points else "",
                    "num_events": len(encoded_labels),
                    "sequence": format_numeric_sequence(clusters),
                    "encoded_labels": format_list(encoded_labels),
                    "anom_sequence": format_list(anomaly_flags),
                    "anomaly_count": anomaly_count,
                    "final_status": final_status,
                    "round_1": format_list(round_1),
                    "round_2": format_list(round_2),
                    "api_error1": format_list(api_round_1),
                    "api_error2": format_list(api_round_2),
                    "assertion_error_round_1": format_list(assertions_round_1),
                    "assertion_error_round_2": format_list(assertions_round_2),
                    "services": format_list(services),
                    "levels": format_list(levels),
                    "content_joined": " ".join(contents),
                }
            )
    return pd.DataFrame.from_records(records)


def process_interval(interval: str) -> None:
    interval = interval.strip()
    input_path = RESOURCES_ROOT / f"extracted_sequences_{interval}.pickle"
    if not input_path.exists():
        raise FileNotFoundError(
            f"Missing {input_path}. Run clog/1_preprocess_data.py with TIME_INTERVAL={interval} first."
        )

    with input_path.open("rb") as handle:
        data = pickle.load(handle)

    df = derive_records(data, interval)
    if df.empty:
        raise ValueError(f"No records were produced for interval {interval} from {input_path}.")

    output_dir = RESOURCES_ROOT / interval
    output_dir.mkdir(parents=True, exist_ok=True)

    normal_df = df[df["anomaly_count"] == 0].reset_index(drop=True)
    abnormal_df = df[df["anomaly_count"] > 0].reset_index(drop=True)

    normal_path = output_dir / f"normal_sequences{interval}.csv"
    abnormal_path = output_dir / f"abnormal_sequences{interval}.csv"

    normal_df.to_csv(normal_path, index=False)
    abnormal_df.to_csv(abnormal_path, index=False)

    print(
        f"Interval {interval}: wrote {len(normal_df)} normal sequences to {normal_path} and "
        f"{len(abnormal_df)} abnormal sequences to {abnormal_path}."
    )


def collect_intervals(requested: Iterable[str]) -> List[str]:
    intervals = [item for item in requested if item]
    if intervals:
        return intervals

    detected = sorted(
        {
            path.stem.replace("extracted_sequences_", "")
            for path in RESOURCES_ROOT.glob("extracted_sequences_*.pickle")
        }
    )
    if not detected:
        raise FileNotFoundError(
            "No extracted_sequences_<interval>.pickle files were found. Run clog/1_preprocess_data.py first "
            "or provide the intervals explicitly."
        )
    return detected


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate normal and abnormal sequence datasets from extracted sequence pickles."
    )
    parser.add_argument(
        "intervals",
        nargs="*",
        help="Time intervals to process (e.g. 300s 600s). If omitted, process every extracted_sequences_<interval>.pickle",
    )
    args = parser.parse_args()

    intervals = collect_intervals(args.intervals)
    for interval in intervals:
        process_interval(interval)


if __name__ == "__main__":
    main()
