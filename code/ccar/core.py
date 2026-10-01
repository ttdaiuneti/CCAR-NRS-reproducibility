"""Compute CORE_delta via dependency drop, without a |U|^2 x |C| structure."""

from __future__ import annotations

import numpy as np

from ccar.nrs import row_blocks


def compute_core_pos(
    X: np.ndarray,
    y: np.ndarray,
    delta: float,
) -> tuple[list[int], float, np.ndarray]:
    """CORE = {a : gamma(C\\{a}) < gamma(C)} together with gamma(C) and POS_C.

    Theorem 3.4 (iii): a is in CORE iff some x in POS_C leaves the positive
    region when a is removed. Under the max-metric,
    Delta_{C\\{a}}(x_i, x_j) = max_{b != a} |b(x_i) - b(x_j)|, which equals the
    second-largest per-attribute difference when a attains the largest and the
    largest otherwise. Keeping only (largest, its attribute, second largest)
    per pair gives every Delta_{C\\{a}} at O(|U|^2 |C|) time and O(block * |U|)
    memory; the per-attribute differences are never stored together.

    Returns (core_indices, gamma_C, pos_C mask).
    """
    y = np.asarray(y)
    n, n_attr = X.shape
    pos = np.empty(n, dtype=bool)
    leaves = np.zeros(n_attr, dtype=bool)  # a removes some x from POS_C
    for s, e in row_blocks(n):
        m1 = np.zeros((e - s, n))
        m2 = np.zeros((e - s, n))
        arg = np.full((e - s, n), -1, dtype=np.int32)
        for a in range(n_attr):
            col = X[:, a]
            d = np.abs(col[s:e, None] - col[None, :])
            top = d > m1
            np.copyto(m2, np.where(top, m1, np.maximum(m2, d)))
            np.copyto(m1, d, where=top)
            arg[top] = a
        diff = y[s:e, None] != y[None, :]
        pos_blk = ~((m1 <= delta) & diff).any(axis=1)
        pos[s:e] = pos_blk
        if not pos_blk.any():
            continue
        # Only pairs (i in POS_C, j cross-class) with m2 <= delta < m1 can make
        # x_i leave POS when the arg-max attribute is removed.
        cand = pos_blk[:, None] & diff & (m1 > delta) & (m2 <= delta)
        leaves[np.unique(arg[cand])] = True
    core = [int(a) for a in np.flatnonzero(leaves)]
    return core, float(pos.sum()) / n if n else 0.0, pos


def compute_core(
    X: np.ndarray,
    y: np.ndarray,
    delta: float,
    attr_names: list[str] | None = None,
) -> tuple[list[int], float]:
    """Returns (core_indices, gamma_C); see compute_core_pos."""
    core, gamma_c, _ = compute_core_pos(X, y, delta)
    return core, gamma_c
