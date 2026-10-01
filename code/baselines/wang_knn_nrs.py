"""Wang-style k-nearest neighborhood rough set greedy reduction.

Reference spirit: Wang et al., IJAR 2019 (k-nearest neighborhood rough sets).
Adapted to this repository's shared Min–Max preprocessing and gamma_delta
scoring so that the method is comparable under the unified protocol: neighborhoods
are the k nearest neighbors under the Chebyshev metric (excluding self), and
dependency / significance use the same POS definition as ccar.nrs with those
neighborhoods substituted for delta-balls.

This is a protocol-matched reimplementation for comparative experiments, not a
line-by-line reproduction of every variant in Wang et al. (2019).
"""

from __future__ import annotations

import numpy as np

from ccar.nrs import normalize_features


def _knn_neighborhoods(
    X: np.ndarray,
    attr_idx: np.ndarray,
    k: int,
) -> list[np.ndarray]:
    """For each i, indices of k nearest neighbors under max-metric on attr_idx."""
    n = X.shape[0]
    if attr_idx.size == 0:
        # Empty attribute set: every other object is a neighbor; take all.
        return [np.array([j for j in range(n) if j != i], dtype=int) for i in range(n)]
    Xb = X[:, attr_idx]
    nbhs: list[np.ndarray] = []
    k_eff = min(k, n - 1)
    for i in range(n):
        diffs = np.max(np.abs(Xb - Xb[i]), axis=1)
        diffs[i] = np.inf
        nn = np.argpartition(diffs, k_eff)[:k_eff]
        nbhs.append(nn)
    return nbhs


def knn_dependency(
    X: np.ndarray,
    y: np.ndarray,
    attr_idx: np.ndarray,
    k: int,
) -> float:
    """gamma_k(B,D) = |POS| / |U| under k-nearest neighborhoods."""
    n = X.shape[0]
    if n == 0:
        return 0.0
    nbhs = _knn_neighborhoods(X, attr_idx, k)
    pos = 0
    for i, nbh in enumerate(nbhs):
        if nbh.size == 0 or np.all(y[nbh] == y[i]):
            pos += 1
    return pos / n


def wang_knn_nrs_reduction(
    X: np.ndarray,
    y: np.ndarray,
    delta: float = 0.1,
    *,
    k: int = 3,
    normalize: bool = True,
) -> list[int]:
    """Forward greedy reduction under k-nearest neighborhood dependency.

    ``delta`` is accepted for API compatibility with other reducers but unused:
    the neighborhood is defined by k, not by a radius.
    """
    del delta
    X_work = normalize_features(X) if normalize else X.astype(float)
    y = np.asarray(y)
    n_attr = X_work.shape[1]
    if n_attr == 0:
        return []

    all_idx = np.arange(n_attr, dtype=int)
    gamma_c = knn_dependency(X_work, y, all_idx, k)
    selected: list[int] = []
    sel_idx = np.array([], dtype=int)

    while knn_dependency(X_work, y, sel_idx, k) < gamma_c - 1e-12:
        remaining = [a for a in range(n_attr) if a not in selected]
        if not remaining:
            break
        best_a, best_g = None, -1.0
        for a in remaining:
            g = knn_dependency(X_work, y, np.array(selected + [a], dtype=int), k)
            if g > best_g:
                best_g, best_a = g, a
        if best_a is None:
            break
        # stall detection
        cur = knn_dependency(X_work, y, sel_idx, k)
        if best_g <= cur + 1e-12:
            break
        selected.append(best_a)
        sel_idx = np.array(selected, dtype=int)

    if not selected:
        # fallback: single best attribute by individual dependency
        scores = [knn_dependency(X_work, y, np.array([a]), k) for a in range(n_attr)]
        selected = [int(np.argmax(scores))]
    return sorted(selected)
