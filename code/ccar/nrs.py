"""Neighborhood rough set primitives (Hu-style max-metric).

Every method in the benchmark (CCAR variants and all baselines) evaluates
neighborhoods through the blocked kernels below, so wall-clock differences
reflect the algorithms rather than differently optimised gamma routines.
Distances are computed one row block and one attribute at a time: memory is
O(block * |U|), and no |U| x |U| x |C| structure is ever materialised.
"""

from __future__ import annotations

from typing import Iterator

import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Rows per block chosen so that one block of pairwise values holds about
# 4M float64 entries (~32 MB) regardless of |U|.
_BLOCK_ENTRIES = 1 << 22


def normalize_features(X: np.ndarray) -> np.ndarray:
    """Scale features to [0, 1] for fixed delta semantics."""
    if X.size == 0:
        return X
    scaler = MinMaxScaler()
    return scaler.fit_transform(X.astype(float))


def row_blocks(n: int) -> Iterator[tuple[int, int]]:
    """Yield [start, stop) row ranges covering 0..n-1."""
    step = max(1, _BLOCK_ENTRIES // max(n, 1))
    for s in range(0, n, step):
        yield s, min(n, s + step)


def block_distance(X: np.ndarray, s: int, e: int, attr_idx: np.ndarray) -> np.ndarray:
    """Max-metric distance Delta_B between rows s..e-1 and all rows: shape (e-s, n)."""
    n = X.shape[0]
    d = np.zeros((e - s, n))
    for a in attr_idx:
        col = X[:, a]
        np.maximum(d, np.abs(col[s:e, None] - col[None, :]), out=d)
    return d


def delta_B(X: np.ndarray, i: int, j: int, attr_idx: np.ndarray) -> float:
    """max_{a in B} |x_i^a - x_j^a|."""
    if attr_idx.size == 0:
        return 0.0
    diff = np.abs(X[i, attr_idx] - X[j, attr_idx])
    return float(np.max(diff))


def neighborhood_indices(
    X: np.ndarray,
    i: int,
    attr_idx: np.ndarray,
    delta: float,
) -> np.ndarray:
    """Indices y with Delta_B(x_i, x_y) <= delta."""
    n = X.shape[0]
    if attr_idx.size == 0:
        return np.arange(n)
    diffs = np.max(np.abs(X[:, attr_idx] - X[i, attr_idx]), axis=1)
    return np.where(diffs <= delta)[0]


def positive_region(
    X: np.ndarray,
    y: np.ndarray,
    attr_idx: np.ndarray,
    delta: float,
) -> np.ndarray:
    """Boolean mask: x in POS_{B,delta}(D)."""
    n = X.shape[0]
    y = np.asarray(y)
    attr_idx = np.asarray(attr_idx, dtype=int)
    if attr_idx.size == 0:
        # Every object neighbours every other object.
        return np.full(n, np.unique(y).size <= 1)
    pos = np.empty(n, dtype=bool)
    for s, e in row_blocks(n):
        conflict = (block_distance(X, s, e, attr_idx) <= delta) & (y[s:e, None] != y[None, :])
        pos[s:e] = ~conflict.any(axis=1)
    return pos


def dependency_degree(
    X: np.ndarray,
    y: np.ndarray,
    attr_idx: np.ndarray,
    delta: float,
) -> float:
    """gamma_delta(B, D) = |POS| / |U|."""
    n = X.shape[0]
    if n == 0:
        return 0.0
    pos = positive_region(X, y, attr_idx, delta)
    return float(np.sum(pos)) / n


def attribute_significance(
    X: np.ndarray,
    y: np.ndarray,
    base_idx: np.ndarray,
    attr: int,
    delta: float,
) -> float:
    """sig_delta(a | B) = gamma(B union {a}) - gamma(B)."""
    g0 = dependency_degree(X, y, base_idx, delta)
    g1 = dependency_degree(X, y, np.append(base_idx, attr), delta)
    return g1 - g0


def backward_prune(
    X: np.ndarray,
    y: np.ndarray,
    selected: list[int],
    removable: list[int],
    gamma_target: float,
    delta: float,
) -> list[int]:
    """Single pass: drop each attribute of `removable` (in the given order) whose
    removal keeps gamma_delta >= gamma_target. The result is minimal with
    respect to `removable`: re-adding a dropped attribute is never needed, and
    by monotonicity (Lemma 3.2) no kept attribute can be dropped afterwards."""
    sel = list(selected)
    for a in removable:
        trial = [b for b in sel if b != a]
        if trial and dependency_degree(X, y, np.array(trial, dtype=int), delta) >= gamma_target - 1e-12:
            sel = trial
    return sorted(sel)
