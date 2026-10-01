"""Ensure reduct reaches gamma(C,D) (approximate reduct repair)."""

from __future__ import annotations

import numpy as np

from ccar.nrs import attribute_significance, dependency_degree


def repair_to_gamma(
    X: np.ndarray,
    y: np.ndarray,
    selected: list[int],
    gamma_c: float,
    delta: float,
    *,
    force_repair: bool = False,
) -> list[int]:
    """Verify gamma(selected) >= gamma_c; optionally repair if force_repair=True.
    
    By default (force_repair=False), this is a sanity check: if gamma < gamma_c,
    a warning is issued but no attributes are added. This preserves the theoretical
    guarantee that the hitting set solution from Theorem 2 should already achieve gamma_c.
    
    Args:
        force_repair: If True, greedily add attributes until gamma >= gamma_c.
                     If False (default), only verify and warn if check fails.
    """
    sel = list(selected)
    idx = np.array(sel, dtype=int) if sel else np.array([], dtype=int)
    n_attr = X.shape[1]
    
    current_gamma = dependency_degree(X, y, idx, delta)
    
    if current_gamma >= gamma_c - 1e-12:
        return sorted(sel)
    
    if not force_repair:
        import warnings
        warnings.warn(
            f"Sanity check failed: gamma({sel})={current_gamma:.6f} < gamma_c={gamma_c:.6f}. "
            "This may indicate a bug in the hitting set or core computation. "
            "Set force_repair=True to attempt automatic repair.",
            RuntimeWarning,
        )
        return sorted(sel)
    
    # Force repair mode: greedily add attributes
    while dependency_degree(X, y, idx, delta) < gamma_c - 1e-12:
        remaining = [a for a in range(n_attr) if a not in sel]
        if not remaining:
            break
        best = max(
            remaining,
            key=lambda a: attribute_significance(X, y, idx, a, delta),
        )
        sel.append(best)
        idx = np.array(sel, dtype=int)
    return sorted(sel)
