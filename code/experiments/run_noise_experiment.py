"""Run noise robustness experiment on small UCI datasets."""

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
DELTA_DEFAULT = 0.1

# Small datasets for noise experiment
SMALL_DATASETS = ["wine", "iris", "glass", "sonar", "ionosphere", "heart"]

ALGORITHMS: dict[str, str] = {
    "ccar_exact": "CCAR-E",
    "ccar_h": "CCAR-H",
    "ccar_surrogate": "Surr.",
    "core_init_greedy": "Core-init",
}


def add_label_noise(y: np.ndarray, noise_ratio: float, random_state: int = 42) -> np.ndarray:
    """Add random label noise by flipping a fraction of labels."""
    rng = np.random.RandomState(random_state)
    n_samples = len(y)
    n_noise = int(n_samples * noise_ratio)
    noise_indices = rng.choice(n_samples, n_noise, replace=False)
    y_noisy = y.copy()
    unique_classes = np.unique(y)
    for idx in noise_indices:
        current_label = y[idx]
        other_classes = unique_classes[unique_classes != current_label]
        if len(other_classes) > 0:
            y_noisy[idx] = rng.choice(other_classes)
    return y_noisy


def add_feature_noise(X: np.ndarray, noise_ratio: float, random_state: int = 42) -> np.ndarray:
    """Add Gaussian noise to a fraction of features."""
    rng = np.random.RandomState(random_state)
    X_noisy = X.copy()
    n_samples, n_features = X.shape
    n_noise_features = int(n_features * noise_ratio)
    noise_feature_indices = rng.choice(n_features, n_noise_features, replace=False)
    for feat_idx in noise_feature_indices:
        noise = rng.normal(0, X[:, feat_idx].std() * 0.5, n_samples)
        X_noisy[:, feat_idx] += noise
    return X_noisy


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


def benchmark_noise_experiment(
    info: DatasetInfo,
    delta: float,
    noise_type: str,
    noise_ratio: float,
    *,
    n_splits: int = N_SPLITS,
) -> list[dict[str, Any]]:
    """Run noise experiment on one dataset."""
    X, y = info.loader()
    n, p = X.shape
    rows: list[dict[str, Any]] = []

    # Add noise
    if noise_type == "label":
        y_noisy = add_label_noise(y, noise_ratio, RANDOM_STATE)
        X_noisy = X
    elif noise_type == "feature":
        X_noisy = add_feature_noise(X, noise_ratio, RANDOM_STATE)
        y_noisy = y
    else:
        X_noisy = X
        y_noisy = y

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    folds = list(skf.split(X_noisy, y_noisy))

    for algo in ALGORITHMS.keys():
        fold_reduct: list[int] = []
        fold_runtime: list[float] = []
        fold_knn: list[float] = []
        fold_svm: list[float] = []

        for train_idx, test_idx in folds:
            X_tr, y_tr = X_noisy[train_idx], y_noisy[train_idx]
            X_te, y_te = X_noisy[test_idx], y_noisy[test_idx]
            
            reduct, rt = run_reduction(algo, X_tr, y_tr, delta)
            fold_reduct.append(len(reduct))
            fold_runtime.append(rt)
            
            # Test on CLEAN data (not noisy)
            scores = fold_test_scores(X[train_idx], y[train_idx], X[test_idx], y[test_idx], reduct)
            fold_knn.append(scores["knn"])
            fold_svm.append(scores["svm"])

        rows.append(
            {
                "dataset": info.key,
                "dataset_name": info.display_name,
                "noise_type": noise_type,
                "noise_ratio": noise_ratio,
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
    parser.add_argument("--noise-type", choices=["label", "feature", "none"], default="label")
    parser.add_argument("--noise-ratio", type=float, default=0.1)
    parser.add_argument("--results-dir", type=Path, default=None)
    args = parser.parse_args()

    results_dir = args.results_dir or Path(__file__).resolve().parents[2] / "results_noise"
    results_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    
    for key in SMALL_DATASETS:
        for info in BENCHMARK_DATASETS:
            if info.key == key:
                print(f"Running {info.display_name} ({key}) with {args.noise_type} noise {args.noise_ratio:.0%}...")
                rows = benchmark_noise_experiment(
                    info, args.delta, args.noise_type, args.noise_ratio
                )
                all_rows.extend(rows)
                break

    # Write results
    out_file = results_dir / f"noise_{args.noise_type}_{args.noise_ratio:.0%}.csv"
    with out_file.open("w", newline="") as f:
        if all_rows:
            w = csv.DictWriter(f, fieldnames=all_rows[0].keys())
            w.writeheader()
            w.writerows(all_rows)
    
    print(f"Wrote {out_file}")
    
    # Print summary
    print("\nSummary (|R|, kNN accuracy):")
    for row in all_rows:
        print(f"{row['algorithm_label']:>10} |R|={row['n_reduct_mean']:.2f} kNN={row['knn_mean']:.3f}")


if __name__ == "__main__":
    main()
