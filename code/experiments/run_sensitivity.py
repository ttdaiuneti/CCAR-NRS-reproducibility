"""Run δ sensitivity analysis on 4 smallest datasets."""

from pathlib import Path
import csv
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier

from experiments.datasets import BENCHMARK_DATASETS
from ccar.ccar_exact import ccar_nrs_exact
from ccar.ccar_heuristic import ccar_nrs_heuristic
from ccar.ccar_surrogate import ccar_nrs_surrogate
from ccar.core import compute_core

SMALL_DATASETS = ["iris", "wine", "sonar", "heart"]
DELTA_VALUES = [0.05, 0.10, 0.15, 0.20]
ALGORITHMS = ["ccar_exact", "ccar_h", "ccar_surrogate"]
RANDOM_STATE = 42
N_SPLITS = 5


def fold_test_scores(X_train, y_train, X_test, y_test, reduct):
    """Compute kNN accuracy on a single fold."""
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train[:, reduct], y_train)
    return {"knn": knn.score(X_test[:, reduct], y_test)}


def run_sensitivity():
    """Run sensitivity analysis for δ on small datasets."""
    results = []
    
    for ds_key in SMALL_DATASETS:
        ds_info = next(d for d in BENCHMARK_DATASETS if d.key == ds_key)
        X, y = ds_info.loader()
        n, p, k = X.shape[0], X.shape[1], len(np.unique(y))
        
        print(f"\nProcessing {ds_key}: n={n}, p={p}, k={k}")
        
        for delta in DELTA_VALUES:
            print(f"  δ={delta}")
            
            # Compute core
            core_attrs, gamma = compute_core(X, y, delta)
            n_core = len(core_attrs)
            
            # Run each algorithm
            for algo in ALGORITHMS:
                if algo == "ccar_exact":
                    res = ccar_nrs_exact(X, y, delta)
                elif algo == "ccar_h":
                    res = ccar_nrs_heuristic(X, y, delta)
                elif algo == "ccar_surrogate":
                    res = ccar_nrs_surrogate(X, y, delta)
                
                reduct = res.reduct_indices
                n_reduct = len(reduct)
                
                # 5-fold CV accuracy
                skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
                knn_scores = []
                for train_idx, test_idx in skf.split(X, y):
                    X_tr, y_tr = X[train_idx], y[train_idx]
                    X_te, y_te = X[test_idx], y[test_idx]
                    s = fold_test_scores(X_tr, y_tr, X_te, y_te, reduct)
                    knn_scores.append(s["knn"])
                
                acc_mean = float(np.mean(knn_scores))
                acc_std = float(np.std(knn_scores))
                
                results.append({
                    "dataset": ds_key,
                    "delta": delta,
                    "algorithm": algo,
                    "n_core": n_core,
                    "n_reduct": n_reduct,
                    "knn_mean": acc_mean,
                    "knn_std": acc_std,
                    "gamma": gamma,
                })
                
                print(f"    {algo}: |R|={n_reduct}, acc={acc_mean:.3f}±{acc_std:.3f}")
    
    # Save results
    root = Path(__file__).resolve().parents[2]
    results_dir = root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = results_dir / "sensitivity_delta.csv"
    
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\nSaved results to {output_path}")
    return results


if __name__ == "__main__":
    run_sensitivity()
