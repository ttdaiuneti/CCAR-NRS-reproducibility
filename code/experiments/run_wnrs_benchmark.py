"""Run WNRS baseline on 18 UCI datasets (5-fold CV); merge into benchmark CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from baselines.wnrs import wnrs_reduction
from ccar.nrs import dependency_degree, normalize_features
from experiments.datasets import BENCHMARK_DATASETS, DatasetInfo

RANDOM_STATE = 42
N_SPLITS = 5
DELTA = 0.1

DATASETS_18 = [
    "wine",
    "sonar",
    "ionosphere",
    "heart",
    "iris",
    "glass",
    "vehicle",
    "ecoli",
    "dermatology",
    "wdbc",
    "credit",
    "diabetes",
    "vowel",
    "australian",
    "chess",
    "vote",
    "yeast",
    "segment",
]


def fold_test_scores(X_train, y_train, X_test, y_test, reduct):
    if not reduct:
        return {"knn": 0.0, "svm": 0.0}
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train[:, reduct])
    X_te_s = scaler.transform(X_test[:, reduct])
    knn = KNeighborsClassifier(n_neighbors=3).fit(X_tr_s, y_train).score(X_te_s, y_test)
    svm = SVC(kernel="rbf", C=1.0, gamma="scale").fit(X_tr_s, y_train).score(X_te_s, y_test)
    return {"knn": float(knn), "svm": float(svm)}


def benchmark_one(info: DatasetInfo, delta: float) -> dict:
    import time

    X, y = info.loader()
    n, p = X.shape
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    fold_reduct, fold_rt, fold_knn, fold_svm, fold_valid = [], [], [], [], []

    for train_idx, test_idx in skf.split(X, y):
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_te, y_te = X[test_idx], y[test_idx]
        Xn = normalize_features(X_tr)
        t0 = time.perf_counter()
        reduct = wnrs_reduction(X_tr, y_tr, delta)
        rt = time.perf_counter() - t0
        gamma_c = dependency_degree(Xn, y_tr, np.arange(p), delta)
        gamma_r = (
            dependency_degree(Xn, y_tr, np.array(reduct, dtype=int), delta)
            if reduct
            else 0.0
        )
        fold_reduct.append(len(reduct))
        fold_rt.append(rt)
        scores = fold_test_scores(X_tr, y_tr, X_te, y_te, reduct)
        fold_knn.append(scores["knn"])
        fold_svm.append(scores["svm"])
        fold_valid.append(gamma_r >= gamma_c - 1e-9 and len(reduct) > 0)

    return {
        "dataset": info.key,
        "dataset_name": info.display_name,
        "n_samples": n,
        "n_features": p,
        "gamma_full": "",
        "algorithm": "wnrs",
        "algorithm_label": "WNRS",
        "n_reduct_mean": float(np.mean(fold_reduct)),
        "n_reduct_std": float(np.std(fold_reduct)),
        "runtime_mean": float(np.mean(fold_rt)),
        "runtime_std": float(np.std(fold_rt)),
        "knn_mean": float(np.mean(fold_knn)),
        "knn_std": float(np.std(fold_knn)),
        "svm_mean": float(np.mean(fold_svm)),
        "svm_std": float(np.std(fold_svm)),
        "n_core": "",
        "n_pairs": "",
        "pair_ratio": "",
        "used_surrogate": "False",
        "valid_reduct": str(all(fold_valid)),
        "t_full_matrix": "",
        "n_matrix_entries": "",
        "speedup_pair_vs_full": "",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--delta", type=float, default=DELTA)
    parser.add_argument("--datasets", nargs="*", default=DATASETS_18)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    results = root / "results"
    key_set = set(args.datasets)
    infos = [d for d in BENCHMARK_DATASETS if d.key in key_set]

    wnrs_rows = []
    for info in infos:
        print(f"WNRS: {info.key}...")
        wnrs_rows.append(benchmark_one(info, args.delta))
        print(
            f"  |R|={wnrs_rows[-1]['n_reduct_mean']:.1f}, "
            f"kNN={wnrs_rows[-1]['knn_mean']:.3f}, "
            f"valid={wnrs_rows[-1]['valid_reduct']}"
        )

    wnrs_path = results / "benchmark_wnrs.csv"
    with wnrs_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(wnrs_rows[0].keys()))
        w.writeheader()
        w.writerows(wnrs_rows)
    print(f"Wrote {wnrs_path}")

    merged_in = results / "benchmark_merged_with_wnrs.csv"
    if not merged_in.is_file():
        merged_in = results / "benchmark_merged.csv"
    if not merged_in.is_file():
        merged_in = results / "benchmark_20datasets.csv"
    base = [r for r in csv.DictReader(merged_in.open()) if r["algorithm"] != "wnrs"]
    base = [
        r
        for r in base
        if r["dataset"] in key_set and r["dataset"] not in ("splice", "spambase")
    ]

    new_path = results / "benchmark_18_full.csv"
    merged = base + wnrs_rows
    with new_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(merged[0].keys()))
        w.writeheader()
        w.writerows(merged)
    print(f"Wrote {new_path} ({len(merged)} rows)")

    # Canonical file for paper tables
    canonical = results / "benchmark_merged_with_wnrs.csv"
    canonical.write_text(new_path.read_text())
    print(f"Updated {canonical}")


if __name__ == "__main__":
    main()
