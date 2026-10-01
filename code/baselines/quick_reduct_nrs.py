"""
QuickReduct-NRS: Forward greedy attribute reduction from empty set
Based on classical QuickReduct algorithm adapted for NRS.
"""

import numpy as np
from typing import List


def nrs_dependency(X: np.ndarray, y: np.ndarray, attrs: List[int], delta: float) -> float:
    """Compute NRS dependency γ_δ(B, D).

    Uses the shared NRS primitives (MinMax scaling + Hu max-metric), identical to
    the definition used by CCAR, WNRS, Core-init and by the reduct-validity check
    in run_benchmark.py.

    Prior to 2026-08 this function used StandardScaler + Euclidean distance, a
    different neighborhood definition from the one the paper states and from the
    one used to score validity. QuickReduct therefore terminated at
    gamma(R) = gamma(C) = 1.0 under its own metric and was then judged invalid
    under the max-metric, producing a spurious 1/18 validity rate. Comparing
    reducts across two different neighborhood definitions is not a valid
    benchmark; all methods must share one definition.
    """
    if not attrs:
        return 0.0

    from ccar.nrs import dependency_degree

    # X is Min-Max normalised once by quick_reduct_nrs (idempotent on [0,1] data).
    return dependency_degree(X, y, np.asarray(attrs, dtype=int), delta)


def quick_reduct_nrs(X: np.ndarray, y: np.ndarray, delta: float, verbose: bool = False) -> List[int]:
    """
    QuickReduct-NRS: Forward greedy from empty set.
    
    Algorithm:
    1. Start with R = ∅
    2. While γ(R, D) < γ(C, D):
       - Find attribute a that maximizes γ(R ∪ {a}, D)
       - Add a to R
    3. Return R
    """
    import time
    t_start = time.time()
    
    # Normalise once here; before 2026-09 nrs_dependency re-normalised X on
    # every gamma call (idempotent, but repeated work charged to QuickReduct).
    from ccar.nrs import normalize_features
    X = normalize_features(X)
    C = list(range(X.shape[1]))
    gamma_C = nrs_dependency(X, y, C, delta)
    
    if verbose:
        print(f"[QuickReduct-NRS] γ(C,D) = {gamma_C:.4f}")
    
    R = []
    
    while True:
        # Check if current R achieves target dependency
        gamma_R = nrs_dependency(X, y, R, delta) if R else 0.0
        
        if abs(gamma_R - gamma_C) < 1e-10:
            break
        
        # Find best attribute to add
        best_attr = None
        best_gamma = -1.0
        
        for a in C:
            if a in R:
                continue
            candidate = R + [a]
            g = nrs_dependency(X, y, candidate, delta)
            if g > best_gamma:
                best_gamma = g
                best_attr = a
        
        if best_attr is None:
            break
        
        R.append(best_attr)
        
        if verbose:
            print(f"  Add attr {best_attr}, γ = {best_gamma:.4f}")
    
    if verbose:
        print(f"Reduct size: {len(R)}, Time: {time.time() - t_start:.4f}s")
    
    return R
