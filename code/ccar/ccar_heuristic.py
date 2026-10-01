"""CCAR-NRS-H: greedy hitting set with significance tie-break, then backward prune."""

from __future__ import annotations

import numpy as np

from ccar.ccar_surrogate import ccar_pipeline
from ccar.nrs import normalize_features
from ccar.result import CCARResult


def ccar_nrs_heuristic(
    X: np.ndarray,
    y: np.ndarray,
    delta: float = 0.1,
    *,
    normalize: bool = True,
) -> CCARResult:
    X_work = normalize_features(X) if normalize else X.astype(float)
    return ccar_pipeline(X_work, y, delta, exact_hitting=False)
