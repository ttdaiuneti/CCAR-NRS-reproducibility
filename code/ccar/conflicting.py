"""Conflicting pairs P_delta and projected family F for M_RES (Theorem 3.9).

Per-attribute NRS discernibility (Hu 2008): a in m_ij^delta iff
|a(x_i) - a(x_j)| > delta. Hence the family entries use threshold delta,
NOT a numerical tolerance.

On an inconsistent table (gamma_delta(C,D) < 1) only pairs with an endpoint in
POS_{C,delta}(D) constrain an approximate reduct (Lemma A.1, M_POS); pass the
POS_C mask to restrict P_delta accordingly. Without it every cross-class pair
in N^CORE is kept, which over-constrains the hitting set on inconsistent tables.
"""

from __future__ import annotations

import numpy as np

from ccar.nrs import block_distance, row_blocks


def conflicting_pairs(
    X: np.ndarray,
    y: np.ndarray,
    core_idx: np.ndarray,
    delta: float,
    pos_mask: np.ndarray | None = None,
) -> np.ndarray:
    """P_delta = {(i,j) : i < j, j in N_delta^CORE(i), y_i != y_j}, optionally
    restricted to pairs with pos_mask[i] or pos_mask[j]. Returns an (m, 2) array
    in row-major (i, then j) order."""
    y = np.asarray(y)
    n = X.shape[0]
    core_idx = np.asarray(core_idx, dtype=int)
    out = []
    cols = np.arange(n)
    for s, e in row_blocks(n):
        mask = (block_distance(X, s, e, core_idx) <= delta) & (y[s:e, None] != y[None, :])
        mask &= cols[None, :] > np.arange(s, e)[:, None]
        if pos_mask is not None:
            mask &= pos_mask[s:e, None] | pos_mask[None, :]
        ii, jj = np.nonzero(mask)
        out.append(np.column_stack([ii + s, jj]))
    return np.concatenate(out) if out else np.empty((0, 2), dtype=int)


def m_res_family(
    X: np.ndarray,
    pairs: np.ndarray,
    core_idx: np.ndarray,
    all_attr_idx: np.ndarray,
    delta: float,
) -> tuple[list[frozenset[int]], list[int]]:
    """F = {{a in C\\CORE : |a(x_i)-a(x_j)| > delta} : (i,j) in P_delta}, non-empty only.

    Identical members are merged: returns the distinct sets in order of first
    occurrence together with their multiplicities, which the hitting-set
    solvers use as weights (so the result equals running them on the full
    multiset).
    """
    non_core = np.setdiff1d(all_attr_idx, core_idx)
    if len(pairs) == 0 or non_core.size == 0:
        return [], []
    rows_all = []
    for s in range(0, len(pairs), 1 << 18):
        p = pairs[s:s + (1 << 18)]
        rows_all.append(np.abs(X[p[:, 0]][:, non_core] - X[p[:, 1]][:, non_core]) > delta)
    B = np.concatenate(rows_all)
    B = B[B.any(axis=1)]
    if B.shape[0] == 0:
        return [], []
    packed = np.packbits(B, axis=1)
    _, first, counts = np.unique(packed, axis=0, return_index=True, return_counts=True)
    order = np.argsort(first)
    family = [frozenset(int(a) for a in non_core[B[first[k]]]) for k in order]
    return family, [int(counts[k]) for k in order]
