"""CCAR-NRS-Exact."""

from __future__ import annotations

import numpy as np

from ccar.ccar_surrogate import ccar_pipeline
from ccar.nrs import dependency_degree, normalize_features
from ccar.result import CCARResult


def ccar_nrs_exact(
    X: np.ndarray,
    y: np.ndarray,
    delta: float = 0.1,
    *,
    normalize: bool = True,
) -> CCARResult:
    """Core-centric reduct with an exact (branch-and-bound) hitting set;
    falls back to the surrogate seed when the core is empty."""
    X_work = normalize_features(X) if normalize else X.astype(float)
    return ccar_pipeline(X_work, y, delta, exact_hitting=True)


def verify_reduct(
    X: np.ndarray,
    y: np.ndarray,
    reduct: list[int],
    gamma_c: float,
    delta: float,
    *,
    normalize: bool = True,
) -> bool:
    """Check gamma(R,D) >= gamma(C,D)."""
    X_work = normalize_features(X) if normalize else X.astype(float)
    idx = np.array(reduct, dtype=int)
    if idx.size == 0:
        return False
    return dependency_degree(X_work, y, idx, delta) >= gamma_c - 1e-9
