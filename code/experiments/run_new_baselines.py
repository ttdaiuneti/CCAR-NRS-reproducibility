"""Run benchmark only for QuickReduct-NRS and GBNRS on 18 datasets."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from baselines.gbnrs import gbnrs_reduction
from baselines.quick_reduct_nrs import quick_reduct_nrs
from experiments.datasets import BENCHMARK_DATASETS, DatasetInfo

RANDOM_STATE = 42
N_SPLITS = 5
DELTA_DEFAULT = 0.1

ALGORITHMS: dict[str, str] = {
    "quick_reduct": "QuickReduct",
    "gbnrs": "GBNRS",
}


def _reduce_quick_reduct(X: np.ndarray, y: np.ndarray, delta: float) -> list[int]:
    return quick_reduct_nrs(X, y, delta)


def _reduce_gbnrs(X: np.ndarray, y: np.ndarray, delta: float) -> list[int]:
    return gbnrs_reduction(X, y, delta, purity_threshold=0.9)


REDUCERS: dict[str, Callable[..., Any]] = {
    "quick_reduct": _reduce_quick_reduct,
    "gbnrs": _reduce_gbnrs,
}


def run_reduction(
    algo: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    delta: float,
) -> tuple[list[int], float, dict[str, Any]]:
    t0 = time.perf_counter()
    meta: dict[str, Any] = {}
    
    reduct = REDUCERS[algo](X_train, y_train, delta)
    runtime = time.perf_counter() - t0
    
    meta = {
        "n_core": None,
        "n_pairs": None,
        "gamma": None,
        "used_surrogate": False,
        "valid": True,
    }
    
    return reduct, runtime, meta


def fold_test_scores(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    reduct: list[int],
) -> dict[str, float]:
    if not reduct:
        return {"knn": 0.0, "svm": 0.0}
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train[:, reduct])
    X_te_s = scaler.transform(X_test[:, reduct])
    knn = KNeighborsClassifier(n_neighbors=3).fit(X_tr_s, y_train).score(X_te_s, y_test)
    svm = (
        SVC(kernel="rbf", C=1.0, gamma="scale")
        .fit(X_tr_s, y_train)
        .score(X_te_s, y_test)
    )
    return {"knn": float(knn), "svm": float(svm)}


def benchmark_dataset(
    info: DatasetInfo,
    delta: float,
    *,
    n_splits: int = N_SPLITS,
) -> list[dict[str, Any]]:
    X, y = info.loader()
    n, p = X.shape
    rows: list[dict[str, Any]] = []

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    folds = list(skf.split(X, y))

    for algo in ALGORITHMS.keys():
        fold_reduct: list[int] = []
        fold_runtime: list[float] = []
        fold_knn: list[float] = []
        fold_svm: list[float] = []

        for train_idx, test_idx in folds:
            X_tr, y_tr = X[train_idx], y[train_idx]
            X_te, y_te = X[test_idx], y[test_idx]
            
            reduct, rt, meta = run_reduction(algo, X_tr, y_tr, delta)
            fold_reduct.append(len(reduct))
            fold_runtime.append(rt)
            
            scores = fold_test_scores(X_tr, y_tr, X_te, y_te, reduct)
            fold_knn.append(scores["knn"])
            fold_svm.append(scores["svm"])

        rows.append(
            {
                "dataset": info.key,
                "dataset_name": info.display_name,
                "n_samples": n,
                "n_attrs": p,
                "algorithm": algo,
                "algorithm_label": ALGORITHMS[algo],
                "n_reduct_mean": float(np.mean(fold_reduct)),
                "n_reduct_std": float(np.std(fold_reduct)),
                "runtime_mean": float(np.mean(fold_runtime)),
                "runtime_std": float(np.std(fold_runtime)),
                "knn_mean": float(np.mean(fold_knn)),
                "knn_std": float(np.std(fold_knn)),
                "svm_mean": float(np.mean(fold_svm)),
                "svm_std": float(np.std(fold_svm)),
            }
        )

    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--delta", type=float, default=DELTA_DEFAULT)
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--parallel", action="store_true")
    args = parser.parse_args()

    results_dir = args.results_dir or Path(__file__).resolve().parents[2] / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    
    if args.parallel:
        with ProcessPoolExecutor() as executor:
            futures = {
                executor.submit(benchmark_dataset, info, args.delta): info
                for info in BENCHMARK_DATASETS
            }
            for future in as_completed(futures):
                info = futures[future]
                try:
                    rows = future.result()
                    all_rows.extend(rows)
                    print(f"Completed {info.display_name}")
                except Exception as e:
                    print(f"Error on {info.display_name}: {e}")
    else:
        for info in BENCHMARK_DATASETS:
            print(f"Running {info.display_name} ({info.key})...")
            rows = benchmark_dataset(info, args.delta)
            all_rows.extend(rows)

    # Write results
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_file = results_dir / f"benchmark_new_baselines_{timestamp}.csv"
    with out_file.open("w", newline="") as f:
        if all_rows:
            w = csv.DictWriter(f, fieldnames=all_rows[0].keys())
            w.writeheader()
            w.writerows(all_rows)
    
    print(f"\nWrote {out_file}")
    print(f"Total rows: {len(all_rows)}")


if __name__ == "__main__":
    main()
