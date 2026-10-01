"""Ablation: full discernibility construction vs the core-first conflicting-pair path.

Both paths use the same blocked NumPy kernels (ccar.nrs / ccar.conflicting), so
the ratio compares algorithms, not interpreter overhead. Before 2026-09 the
proxy was a pure-Python triple loop while the pair path was vectorised.
"""

from __future__ import annotations

import time

import numpy as np

from ccar.conflicting import conflicting_pairs, m_res_family
from ccar.core import compute_core_pos
from ccar.nrs import normalize_features


def time_pair_only_path(
    X: np.ndarray,
    y: np.ndarray,
    delta: float,
) -> tuple[float, int, int, float]:
    """Core extraction, residual pair identification (M_POS) and F construction.
    Returns (t_sec, |CORE|, |P|, |P|/binom(n,2))."""
    X = normalize_features(X)
    y = np.asarray(y)
    t0 = time.perf_counter()
    core, _, pos = compute_core_pos(X, y, delta)
    core_idx = np.array(core, dtype=int)
    pairs = conflicting_pairs(X, y, core_idx, delta, pos_mask=pos)
    _ = m_res_family(X, pairs, core_idx, np.arange(X.shape[1]), delta)
    elapsed = time.perf_counter() - t0
    u2 = X.shape[0] * (X.shape[0] - 1) / 2
    ratio = len(pairs) / u2 if u2 else 0.0
    return elapsed, len(core), len(pairs), ratio


def time_full_matrix_proxy(
    X: np.ndarray,
    y: np.ndarray,
    delta: float,
) -> tuple[float, int]:
    """Construct every non-empty entry m_ij^delta of M(C,D) over all cross-class
    pairs. Returns (t_sec, number of non-empty entries)."""
    X = normalize_features(X)
    y = np.asarray(y)
    t0 = time.perf_counter()
    empty = np.array([], dtype=int)
    all_pairs = conflicting_pairs(X, y, empty, delta)
    _, weights = m_res_family(X, all_pairs, empty, np.arange(X.shape[1]), delta)
    elapsed = time.perf_counter() - t0
    return elapsed, int(sum(weights))
