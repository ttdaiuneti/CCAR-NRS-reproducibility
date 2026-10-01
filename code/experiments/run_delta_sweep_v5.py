"""Neighborhood-radius (delta) sweep under the v5 unified protocol.

Replaces the pre-2026-08 `run_delta_sensitivity.py`, whose accuracies were
computed on a different (unharmonized) representation and whose gamma column
was taken at a single delta.  Here every quantity -- the reduct, the dependency
target gamma_delta(C,D), the validity check and both classifier accuracies --
is recomputed at each delta on the Min--Max/max-metric representation that
`run_benchmark.py` uses, with the identical StratifiedKFold(5, seed 42) split.
"""

from __future__ import annotations

import csv
import time
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from ccar.ccar_exact import ccar_nrs_exact
from ccar.core import compute_core
from ccar.nrs import dependency_degree, normalize_features
from experiments.datasets import BENCHMARK_DATASETS

RANDOM_STATE = 42
N_SPLITS = 5
DELTAS = [0.05, 0.10, 0.15, 0.20, 0.30]
KEYS = ["iris", "wine", "sonar", "heart", "glass", "ecoli", "diabetes", "australian"]


def main() -> None:
    info = {d.key: d for d in BENCHMARK_DATASETS}
    out = Path(__file__).resolve().parents[2] / "results_v7" / "delta_sweep_v7.csv"
    rows = []
    for key in KEYS:
        X, y = info[key].loader()
        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
        folds = list(skf.split(X, y))
        for delta in DELTAS:
            gamma_full = compute_core(normalize_features(X), y, delta)[1]
            nR, rt, kn, sv, ok = [], [], [], [], []
            for tr, te in folds:
                Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]
                t0 = time.perf_counter()
                res = ccar_nrs_exact(Xtr, ytr, delta)
                rt.append(time.perf_counter() - t0)
                reduct = res.reduct_indices
                nR.append(len(reduct))
                Xn = normalize_features(Xtr)
                idx = np.array(reduct, dtype=int) if reduct else np.array([], dtype=int)
                gr = dependency_degree(Xn, ytr, idx, delta) if reduct else 0.0
                ok.append(bool(reduct) and gr >= res.gamma_c - 1e-9)
                sc = StandardScaler()
                A = sc.fit_transform(Xtr[:, reduct])
                B = sc.transform(Xte[:, reduct])
                kn.append(KNeighborsClassifier(n_neighbors=3).fit(A, ytr).score(B, yte))
                sv.append(SVC(kernel="rbf", C=1.0, gamma="scale").fit(A, ytr).score(B, yte))
            rows.append(
                {
                    "dataset": key,
                    "dataset_name": info[key].display_name,
                    "delta": delta,
                    "gamma_full": gamma_full,
                    "n_reduct_mean": float(np.mean(nR)),
                    "n_reduct_std": float(np.std(nR)),
                    "runtime_mean": float(np.mean(rt)),
                    "knn_mean": float(np.mean(kn)),
                    "knn_std": float(np.std(kn)),
                    "svm_mean": float(np.mean(sv)),
                    "svm_std": float(np.std(sv)),
                    "valid_folds": int(sum(ok)),
                }
            )
            print(
                f"[{key}] delta={delta:.2f} gamma={gamma_full:.3f} "
                f"|R|={np.mean(nR):.1f} knn={np.mean(kn):.3f} valid={sum(ok)}/5",
                flush=True,
            )
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("WROTE", out, len(rows))


if __name__ == "__main__":
    main()
