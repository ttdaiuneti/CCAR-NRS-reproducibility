"""Run delta sensitivity experiment on smallest UCI datasets."""

from __future__ import annotations

import argparse
import csv
import numpy as np
from pathlib import Path
from typing import Any

from ccar.ccar_exact import ccar_nrs_exact
from ccar.ccar_heuristic import ccar_nrs_heuristic
from ccar.ccar_surrogate import ccar_nrs_surrogate
from baselines.greedy_core_init import core_init_greedy
from experiments.datasets import BENCHMARK_DATASETS, DatasetInfo
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

RANDOM_STATE = 42
N_SPLITS = 5

# 4 smallest datasets
SMALL_DATASETS = ["iris", "wine", "glass", "sonar"]

DELTA_VALUES = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]

ALGORITHMS: dict[str, str] = {
    "ccar_exact": "CCAR-E",
    "ccar_h": "CCAR-H",
    "ccar_surrogate": "Surr.",
    "core_init_greedy": "Core-init",
}


def run_reduction(
    algo: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    delta: float,
) -> tuple[list[int], float]:
    """Run reduction algorithm and return reduct and runtime."""
    import time
    t0 = time.perf_counter()
    if algo == "ccar_exact":
        res = ccar_nrs_exact(X_train, y_train, delta)
        reduct = res.reduct_indices
    elif algo == "ccar_h":
        res = ccar_nrs_heuristic(X_train, y_train, delta)
        reduct = res.reduct_indices
    elif algo == "ccar_surrogate":
        res = ccar_nrs_surrogate(X_train, y_train, delta)
        reduct = res.reduct_indices
    elif algo == "core_init_greedy":
        reduct = core_init_greedy(X_train, y_train, delta)
    else:
        raise ValueError(f"Unknown algorithm: {algo}")
    runtime = time.perf_counter() - t0
    return reduct, runtime


def fold_test_scores(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    reduct: list[int],
) -> dict[str, float]:
    """Compute kNN and SVM accuracy on test set."""
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


def benchmark_delta_sensitivity(
    info: DatasetInfo,
    delta: float,
    *,
    n_splits: int = N_SPLITS,
) -> list[dict[str, Any]]:
    """Run delta sensitivity experiment on one dataset."""
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
            
            reduct, rt = run_reduction(algo, X_tr, y_tr, delta)
            fold_reduct.append(len(reduct))
            fold_runtime.append(rt)
            
            scores = fold_test_scores(X_tr, y_tr, X_te, y_te, reduct)
            fold_knn.append(scores["knn"])
            fold_svm.append(scores["svm"])

        rows.append(
            {
                "dataset": info.key,
                "dataset_name": info.display_name,
                "delta": delta,
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
    parser.add_argument("--results-dir", type=Path, default=None)
    args = parser.parse_args()

    results_dir = args.results_dir or Path(__file__).resolve().parents[2] / "results_delta"
    results_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    
    for key in SMALL_DATASETS:
        for info in BENCHMARK_DATASETS:
            if info.key == key:
                for delta in DELTA_VALUES:
                    print(f"Running {info.display_name} ({key}) with delta={delta:.2f}...")
                    rows = benchmark_delta_sensitivity(info, delta)
                    all_rows.extend(rows)
                break

    # Write results
    out_file = results_dir / "delta_sensitivity.csv"
    with out_file.open("w", newline="") as f:
        if all_rows:
            w = csv.DictWriter(f, fieldnames=all_rows[0].keys())
            w.writeheader()
            w.writerows(all_rows)
    
    print(f"\nWrote {out_file}")
    
    # Print summary
    print("\nSummary by delta (CCAR-E |R|, kNN):")
    for delta in DELTA_VALUES:
        delta_rows = [r for r in all_rows if r['delta'] == delta and r['algorithm'] == 'ccar_exact']
        if delta_rows:
            avg_r = np.mean([r['n_reduct_mean'] for r in delta_rows])
            avg_knn = np.mean([r['knn_mean'] for r in delta_rows])
            print(f"delta={delta:.2f}: |R|={avg_r:.2f}, kNN={avg_knn:.3f}")


if __name__ == "__main__":
    main()
