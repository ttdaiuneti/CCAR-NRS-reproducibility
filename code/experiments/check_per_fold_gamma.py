"""Check per-fold γ_δ for Ecoli and Glass."""

from pathlib import Path
import numpy as np
from sklearn.model_selection import StratifiedKFold

from experiments.datasets import BENCHMARK_DATASETS
from ccar.core import compute_core

SMALL_DATASETS = ["ecoli", "glass"]
RANDOM_STATE = 42
N_SPLITS = 5
DELTA = 0.1


def check_per_fold_gamma():
    """Check per-fold γ_δ for Ecoli and Glass."""
    results = {}
    
    for ds_key in SMALL_DATASETS:
        ds_info = next(d for d in BENCHMARK_DATASETS if d.key == ds_key)
        X, y = ds_info.loader()
        n, p, k = X.shape[0], X.shape[1], len(np.unique(y))
        
        print(f"\nProcessing {ds_key}: n={n}, p={p}, k={k}")
        
        # Full-data γ
        core_full, gamma_full = compute_core(X, y, DELTA)
        print(f"  Full-data γ: {gamma_full:.6f}")
        
        # Per-fold γ
        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
        fold_gammas = []
        
        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            X_train, y_train = X[train_idx], y[train_idx]
            core_fold, gamma_fold = compute_core(X_train, y_train, DELTA)
            fold_gammas.append(gamma_fold)
            print(f"  Fold {fold_idx+1}: γ={gamma_fold:.6f}")
        
        mean_gamma = np.mean(fold_gammas)
        print(f"  Mean fold γ: {mean_gamma:.6f}")
        
        results[ds_key] = {
            "full_data_gamma": gamma_full,
            "fold_gammas": fold_gammas,
            "mean_fold_gamma": mean_gamma,
        }
    
    return results


if __name__ == "__main__":
    check_per_fold_gamma()
