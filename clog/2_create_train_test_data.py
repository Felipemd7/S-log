import numpy as np
import pandas as pd
from pathlib import Path

TIME_INTERVAL_ = "180s"

BASE_DIR = Path(__file__).resolve().parents[1]
RESOURCES_DIR = BASE_DIR / "data" / "NOVA" / "resources" / TIME_INTERVAL_
NORMAL_PATH = RESOURCES_DIR / f"normal_sequences{TIME_INTERVAL_}.csv"
ABNORMAL_PATH = RESOURCES_DIR / f"abnormal_sequences{TIME_INTERVAL_}.csv"
SPLIT_DIR = RESOURCES_DIR / "train_valid_test_splits"
SPLIT_DIR.mkdir(parents=True, exist_ok=True)


def _read_sequences(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Expected {label} sequences for interval {TIME_INTERVAL_} at {path}. "
            "Run clog/1_preprocess_data.py followed by clog/create_sequences.py, or copy the prepared CSVs into this folder."
        )
    data = pd.read_csv(path)
    return data.drop(columns=["Unnamed: 0"], errors="ignore")


def filter_(x, list_av):
    return x in list_av


def create_dataset(dataset: pd.DataFrame, valid_experiments) -> pd.DataFrame:
    dataset["to_drop"] = dataset.experiment_id.apply(lambda x: filter_(x, valid_experiments))
    data = dataset[dataset.to_drop].drop(columns=["to_drop"])
    return data


normal_data = _read_sequences(NORMAL_PATH, "normal")
abnormal_data = _read_sequences(ABNORMAL_PATH, "abnormal")

experiments_normal = np.unique(normal_data.experiment_id)
experiments_abnormal = np.unique(abnormal_data.experiment_id)

SIZE_E_NORMAL = len(experiments_normal)
SIZE_E_ABNORMAL = len(experiments_abnormal)

# TRAIN
SIZE_E_NORMAL_TRAIN = int(0.59 * SIZE_E_NORMAL)
SIZE_E_NORMAL_VALID = int(0.5 * (1 - 0.59) * SIZE_E_NORMAL)

train_indices_normal = np.random.choice(experiments_normal, size=SIZE_E_NORMAL_TRAIN, replace=False)
valid_indices_normal = np.random.choice(
    list(set(experiments_normal).difference(set(train_indices_normal))),
    size=SIZE_E_NORMAL_VALID,
    replace=False,
)
test_indices_normal = np.array(
    list(
        set(experiments_normal)
        .difference(set(train_indices_normal))
        .difference(set(valid_indices_normal))
    )
)
assert (
    train_indices_normal.shape[0]
    + valid_indices_normal.shape[0]
    + test_indices_normal.shape[0]
    == SIZE_E_NORMAL
), "Indices are lost"

# VALID
SIZE_E_ABNORMAL_VALID = int(0.5 * SIZE_E_ABNORMAL)
valid_indices_abnormal = np.random.choice(
    experiments_abnormal, size=SIZE_E_ABNORMAL_VALID, replace=False
)
test_indices_abnormal = np.array(
    list(set(experiments_abnormal).difference(set(valid_indices_abnormal)))
)
assert (
    valid_indices_abnormal.shape[0] + test_indices_abnormal.shape[0]
    == SIZE_E_ABNORMAL
), "Indices are lost"


train_normal = create_dataset(normal_data, train_indices_normal)
valid_normal = create_dataset(normal_data, valid_indices_normal)
test_normal = create_dataset(normal_data, test_indices_normal)

valid_abnormal = create_dataset(abnormal_data, valid_indices_abnormal)
test_abnormal = create_dataset(abnormal_data, test_indices_abnormal)

train_normal.to_csv(SPLIT_DIR / f"train_normal_{TIME_INTERVAL_}.csv", index=False)
valid_normal.to_csv(SPLIT_DIR / f"valid_normal_{TIME_INTERVAL_}.csv", index=False)
test_normal.to_csv(SPLIT_DIR / f"test_normal_{TIME_INTERVAL_}.csv", index=False)

valid_abnormal.to_csv(SPLIT_DIR / f"valid_abnormal_{TIME_INTERVAL_}.csv", index=False)
test_abnormal.to_csv(SPLIT_DIR / f"test_abnormal_{TIME_INTERVAL_}.csv", index=False)
