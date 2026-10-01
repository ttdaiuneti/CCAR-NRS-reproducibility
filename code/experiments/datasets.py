"""UCI classification benchmarks for CCAR-NRS (8 original + 10 extended = 18)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import csv

import numpy as np
from sklearn.datasets import load_breast_cancer, load_iris, load_wine
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

DATA_UCI = Path(__file__).resolve().parents[2] / "data" / "uci"

# Stratified caps for very large sets (~15 min full benchmark on laptop-class CPU).
MAX_SAMPLES: dict[str, int] = {
    "chess": 3200,
}


@dataclass(frozen=True)
class DatasetInfo:
    key: str
    display_name: str
    loader: Callable[[], tuple[np.ndarray, np.ndarray]]
    openml_id: int | None = None
    uci_ref: str = ""


def _require(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}. Run: python -m experiments.fetch_uci_data && "
            "python -m experiments.fetch_extended_uci"
        )
    return path


def _cap_samples(
    X: np.ndarray,
    y: np.ndarray,
    key: str,
    *,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    cap = MAX_SAMPLES.get(key)
    if cap is None or X.shape[0] <= cap:
        return X, y
    sss = StratifiedShuffleSplit(n_splits=1, train_size=cap, random_state=seed)
    idx, _ = next(sss.split(X, y))
    return X[idx], y[idx]


def _xy_from_bunch(bunch) -> tuple[np.ndarray, np.ndarray]:
    X = np.asarray(bunch.data, dtype=float)
    y = LabelEncoder().fit_transform(bunch.target)
    return X, y


def _is_number(token: str) -> bool:
    try:
        float(token)
    except (TypeError, ValueError):
        return False
    return True


def _has_header(rows: list[list[str]], *, probe: int = 50) -> bool:
    """Decide whether rows[0] is a column-name line rather than a data record.

    Several of the CSV exports under data/uci/ carry no header line at all
    (glass, ecoli, vowel, heart, german, sonar, ionosphere, splice, ...).  Before
    2026-08 the loader below called next(reader) unconditionally, so for those
    files one genuine instance was silently discarded -- Glass came out at 213
    rows instead of the 214 the UCI reference reports.  Two signals are used:
    a non-numeric token sitting over an all-numeric column, or every nominal
    column carrying a first-row token that never recurs in the data.
    """
    if len(rows) < 3:
        return False
    first, data = rows[0], rows[1 : min(len(rows), 1 + probe)]
    nominal: list[int] = []
    for j in range(len(first)):
        col = [r[j] for r in data if len(r) > j]
        if not col:
            continue
        if all(_is_number(v) for v in col):
            if not _is_number(first[j]):
                return True
        else:
            nominal.append(j)
    if len(nominal) >= 2:
        unseen = sum(
            1 for j in nominal if first[j] not in {r[j] for r in data if len(r) > j}
        )
        if unseen == len(nominal):
            return True
    return False


def _load_openml_csv(path: Path, *, target_col: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    with path.open(newline="") as f:
        all_rows = [r for r in csv.reader(f) if r]
    if not all_rows:
        raise ValueError(f"Empty dataset: {path}")
    if _has_header(all_rows):
        header, rows = all_rows[0], all_rows[1:]
    else:
        header, rows = None, all_rows
    if not rows:
        raise ValueError(f"Empty dataset: {path}")
    ncols = len(rows[0])
    if target_col is None:
        if header and header[0].lower() == "class":
            target_col = 0
        else:
            target_col = -1
    tidx = target_col % ncols
    feat_idx = [i for i in range(ncols) if i != tidx]
    y = LabelEncoder().fit_transform([r[tidx] for r in rows])
    X_raw = np.array([[r[i] for i in feat_idx] for r in rows], dtype=str)
    try:
        X = X_raw.astype(float)
    except ValueError:
        X = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1).fit_transform(
            X_raw
        ).astype(float)
    m = np.isfinite(X).all(axis=1)
    return X[m], y[m]


def _load_csv_numeric(path: Path, *, target_col: int = -1) -> tuple[np.ndarray, np.ndarray]:
    raw = np.genfromtxt(path, delimiter=",", dtype=str, autostrip=True)
    if raw.ndim == 1:
        raw = raw.reshape(1, -1)
    y = LabelEncoder().fit_transform(raw[:, target_col])
    feat_idx = [i for i in range(raw.shape[1]) if i != (target_col % raw.shape[1])]
    X = raw[:, feat_idx].astype(float)
    return X, y


def _make_loader(fn: Callable[[], tuple[np.ndarray, np.ndarray]], key: str):
    def _load() -> tuple[np.ndarray, np.ndarray]:
        X, y = fn()
        return _cap_samples(X, y, key)

    return _load


def _extended(key: str) -> Callable[[], tuple[np.ndarray, np.ndarray]]:
    fname = {
        "glass": "glass.csv",
        "vehicle": "vehicle.csv",
        "vowel": "vowel.csv",
        "australian": "australian.csv",
        "ecoli": "ecoli.csv",
        "yeast": "yeast.csv",
        "dermatology": "dermatology.csv",
        "segment": "segment.csv",
        "splice": "splice.csv",
        "chess": "chess.csv",
        "spambase": "spambase.csv",
        "vote": "vote.csv",
    }[key]

    def _load() -> tuple[np.ndarray, np.ndarray]:
        return _load_openml_csv(_require(DATA_UCI / fname))

    return _load


# --- Original eight ---
def load_wine_uci() -> tuple[np.ndarray, np.ndarray]:
    return _xy_from_bunch(load_wine())


def load_wdbc() -> tuple[np.ndarray, np.ndarray]:
    return _xy_from_bunch(load_breast_cancer())


def load_iris_uci() -> tuple[np.ndarray, np.ndarray]:
    return _xy_from_bunch(load_iris())


def load_sonar() -> tuple[np.ndarray, np.ndarray]:
    path = _require(DATA_UCI / "sonar.csv")
    raw = np.genfromtxt(path, delimiter=",", dtype=str, autostrip=True)
    y = LabelEncoder().fit_transform(raw[:, -1])
    return raw[:, :-1].astype(float), y


def load_ionosphere() -> tuple[np.ndarray, np.ndarray]:
    path = _require(DATA_UCI / "ionosphere.csv")
    raw = np.genfromtxt(path, delimiter=",", dtype=str, autostrip=True)
    y = LabelEncoder().fit_transform(raw[:, -1])
    return raw[:, :-1].astype(float), y


def load_heart() -> tuple[np.ndarray, np.ndarray]:
    path = _require(DATA_UCI / "heart.csv")
    raw = np.genfromtxt(path, delimiter=",", dtype=str, autostrip=True)
    y = (raw[:, -1].astype(float) > 0).astype(int)
    X = np.empty((raw.shape[0], raw.shape[1] - 1), dtype=float)
    for j in range(X.shape[1]):
        col = raw[:, j].copy()
        col[col == "?"] = np.nan
        col = col.astype(float)
        med = np.nanmedian(col)
        col[np.isnan(col)] = med
        X[:, j] = col
    return X, y


def load_german_credit() -> tuple[np.ndarray, np.ndarray]:
    path = _require(DATA_UCI / "german.csv")
    raw = np.genfromtxt(path, delimiter=",", dtype=str, autostrip=True)
    y = LabelEncoder().fit_transform(raw[:, -1])
    X = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1).fit_transform(
        raw[:, :-1]
    ).astype(float)
    return X, y


def load_diabetes() -> tuple[np.ndarray, np.ndarray]:
    return _load_csv_numeric(_require(DATA_UCI / "pima-indians-diabetes.csv"))


EIGHT_DATASETS: list[DatasetInfo] = [
    DatasetInfo("wine", "Wine", _make_loader(load_wine_uci, "wine")),
    DatasetInfo("wdbc", "WDBC", _make_loader(load_wdbc, "wdbc")),
    DatasetInfo("sonar", "Sonar", _make_loader(load_sonar, "sonar")),
    DatasetInfo("ionosphere", "Ionosphere", _make_loader(load_ionosphere, "ionosphere")),
    DatasetInfo("heart", "Heart", _make_loader(load_heart, "heart")),
    DatasetInfo("credit", "German Credit", _make_loader(load_german_credit, "credit")),
    DatasetInfo("diabetes", "Diabetes", _make_loader(load_diabetes, "diabetes")),
    DatasetInfo("iris", "Iris", _make_loader(load_iris_uci, "iris")),
]

EXTENDED_DATASETS: list[DatasetInfo] = [
    DatasetInfo("glass", "Glass", _make_loader(_extended("glass"), "glass"), 40, "Glass"),
    DatasetInfo("vehicle", "Vehicle", _make_loader(_extended("vehicle"), "vehicle"), 54, "Statlog vehicle"),
    DatasetInfo("vowel", "Vowel", _make_loader(_extended("vowel"), "vowel"), 307, "Vowel"),
    DatasetInfo("australian", "Australian", _make_loader(_extended("australian"), "australian"), 40981, "Australian credit"),
    DatasetInfo("ecoli", "Ecoli", _make_loader(_extended("ecoli"), "ecoli"), 39, "Ecoli"),
    DatasetInfo("yeast", "Yeast", _make_loader(_extended("yeast"), "yeast"), 181, "Yeast"),
    DatasetInfo("dermatology", "Dermatology", _make_loader(_extended("dermatology"), "dermatology"), 35, "Dermatology"),
    DatasetInfo("segment", "Segment", _make_loader(_extended("segment"), "segment"), 36, "Image segmentation"),
    # openml_id was 1459 (= artificial-characters) by mistake; KR-vs-KP is did=3.
    DatasetInfo("chess", "Chess (KR-KP)", _make_loader(_extended("chess"), "chess"), 3, "KR vs KP"),
    DatasetInfo("vote", "Congressional", _make_loader(_extended("vote"), "vote"), 105, "House votes"),
]

BENCHMARK_DATASETS: list[DatasetInfo] = EIGHT_DATASETS + EXTENDED_DATASETS
TWENTY_DATASETS = BENCHMARK_DATASETS


def load_dataset(key: str) -> tuple[np.ndarray, np.ndarray]:
    for info in BENCHMARK_DATASETS:
        if info.key == key:
            return info.loader()
    raise KeyError(f"Unknown dataset: {key}")


def dataset_shape(key: str) -> tuple[int, int, int]:
    X, y = load_dataset(key)
    return X.shape[0], X.shape[1], len(np.unique(y))
