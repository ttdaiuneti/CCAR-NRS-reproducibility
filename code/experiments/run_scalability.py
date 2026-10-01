"""Scalability experiment on Covertype (sklearn), n up to 10 000.

Runs CCAR-H, Core-init and QuickReduct on one stratified subsample per size
(seed 42). The timer covers the reduction call only; |CORE|, gamma_delta(C) and
the validity check gamma_delta(R) >= gamma_delta(C) are computed outside it.
Writes results_v7/scalability_v7.csv.
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import StratifiedShuffleSplit

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))

from ccar.nrs import dependency_degree, normalize_features
from ccar.core import compute_core
from ccar.ccar_heuristic import ccar_nrs_heuristic
from baselines.greedy_core_init import core_init_greedy
from baselines.quick_reduct_nrs import quick_reduct_nrs

DELTA = 0.1
SIZES = [1000, 2000, 5000, 10000]
OUT = ROOT / "results_v7" / "scalability_v7.csv"


def subsample(X, y, n, seed=42):
    sss = StratifiedShuffleSplit(n_splits=1, train_size=n, random_state=seed)
    idx, _ = next(sss.split(X, y))
    return X[idx], y[idx]


METHODS = {
    "CCAR-H": lambda X, y: ccar_nrs_heuristic(X, y, DELTA).reduct_indices,
    "Core-init": lambda X, y: core_init_greedy(X, y, DELTA),
    "QuickReduct": lambda X, y: quick_reduct_nrs(X, y, DELTA),
}


def main():
    bunch = fetch_covtype()
    X_all = bunch.data.astype(float)
    y_all = (bunch.target - 1).astype(int)
    rows = []
    for n in SIZES:
        X, y = subsample(X_all, y_all, n)
        Xn = normalize_features(X)
        core, gamma_c = compute_core(Xn, y, DELTA)
        print(f"\nn={n}  |C|={X.shape[1]}  |CORE|={len(core)}  gamma(C)={gamma_c:.4f}", flush=True)
        for name, fn in METHODS.items():
            t0 = time.perf_counter()
            R = fn(X, y)
            rt = time.perf_counter() - t0
            g_r = dependency_degree(Xn, y, np.array(R, dtype=int), DELTA) if R else 0.0
            valid = bool(R) and g_r >= gamma_c - 1e-9
            print(f"  {name:12s} |R|={len(R)}  t={rt:.1f}s  valid={valid}", flush=True)
            rows.append({"n": n, "p": X.shape[1], "n_core": len(core), "gamma_C": gamma_c,
                         "method": name, "n_reduct": len(R), "runtime": rt, "gamma_R": g_r,
                         "valid": valid, "reduct_set": ";".join(map(str, sorted(R)))})
            OUT.parent.mkdir(parents=True, exist_ok=True)
            with OUT.open("w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
    print(f"\nSaved -> {OUT}")


if __name__ == "__main__":
    main()
