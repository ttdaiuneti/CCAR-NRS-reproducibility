"""Experiment pipeline with 5-fold CV, SVM(RBF), 3-NN, and core-init baseline."""

from __future__ import annotations

import csv
import time
from pathlib import Path

import numpy as np
from sklearn.datasets import load_breast_cancer, load_wine, load_iris, load_digits, make_classification
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler

from ccar.ccar_exact import ccar_nrs_exact, verify_reduct
from ccar.ccar_heuristic import ccar_nrs_heuristic
from ccar.ccar_surrogate import ccar_nrs_surrogate
from ccar.core import compute_core
from baselines.greedy_core_init import core_init_greedy
from baselines.gbnrs import gbnrs_reduction

DATASETS = {
    "wine": load_wine,
    "wdbc": load_breast_cancer,
    "iris": load_iris,
    "synthetic1": lambda: make_classification(n_samples=200, n_features=15, n_informative=8, n_redundant=4, random_state=42),
    "synthetic2": lambda: make_classification(n_samples=250, n_features=20, n_informative=10, n_redundant=5, random_state=42),
    "synthetic3": lambda: make_classification(n_samples=300, n_features=25, n_informative=12, n_redundant=6, random_state=42),
    "synthetic4": lambda: make_classification(n_samples=350, n_features=30, n_informative=15, n_redundant=8, random_state=42),
    "synthetic5": lambda: make_classification(n_samples=400, n_features=35, n_informative=18, n_redundant=10, random_state=42),
}


def run_single(
    name: str,
    X: np.ndarray,
    y: np.ndarray,
    delta: float,
    results_dir: Path,
) -> None:
    rows = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for algo, fn in [
        ("ccar_exact", ccar_nrs_exact),
        ("ccar_h", ccar_nrs_heuristic),
        ("ccar_surrogate", ccar_nrs_surrogate),
        ("core_init_greedy", core_init_greedy),
        ("gbnrs", gbnrs_reduction),
    ]:
        t0 = time.perf_counter()
        if algo in ["core_init_greedy", "gbnrs"]:
            reduct = fn(X, y, delta)
            core, gamma_c = compute_core(X, y, delta)
            n_pairs = None
        else:
            res = fn(X, y, delta)
            reduct = res.reduct_indices
            core = res.core_indices
            gamma_c = res.gamma_c
            n_pairs = res.n_pairs
        runtime = time.perf_counter() - t0

        X_sub = X[:, reduct] if reduct else X
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X_sub)

        # 3-NN with 5-fold CV
        knn_scores = cross_val_score(KNeighborsClassifier(n_neighbors=3), Xs, y, cv=skf)
        knn_mean = float(np.mean(knn_scores))
        knn_std = float(np.std(knn_scores))

        # SVM(RBF) with 5-fold CV
        svm_scores = cross_val_score(SVC(kernel='rbf', C=1.0, gamma='scale'), Xs, y, cv=skf)
        svm_mean = float(np.mean(svm_scores))
        svm_std = float(np.std(svm_scores))

        # Accuracy on full dataset (train on full, test on full)
        knn_full = KNeighborsClassifier(n_neighbors=3).fit(Xs, y).score(Xs, y)
        svm_full = SVC(kernel='rbf', C=1.0, gamma='scale').fit(Xs, y).score(Xs, y)

        valid = verify_reduct(X, y, reduct, gamma_c, delta) if algo != "core_init_greedy" else True

        rows.append(
            {
                "dataset": name,
                "algorithm": algo,
                "n_reduct": len(reduct),
                "n_core": len(core),
                "n_pairs": n_pairs,
                "gamma": gamma_c,
                "knn_mean": knn_mean,
                "knn_std": knn_std,
                "knn_full": knn_full,
                "svm_mean": svm_mean,
                "svm_std": svm_std,
                "svm_full": svm_full,
                "runtime_s": runtime,
                "valid_reduct": valid,
            }
        )

    results_dir.mkdir(parents=True, exist_ok=True)
    out = results_dir / f"{name}_summary.csv"
    if rows:
        with out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)
    print(f"Wrote {out}")


def main():
    results_dir = Path(__file__).resolve().parents[2] / "results"
    for name, loader in DATASETS.items():
        data = loader()
        # Handle different data structures from sklearn vs make_classification
        if isinstance(data, tuple):
            X, y = data
        elif hasattr(data, 'data'):
            X = data.data
            y = data.target
        else:
            X, y = data
        y = LabelEncoder().fit_transform(y)
        run_single(name, X, y, delta=0.1, results_dir=results_dir)


if __name__ == "__main__":
    main()
