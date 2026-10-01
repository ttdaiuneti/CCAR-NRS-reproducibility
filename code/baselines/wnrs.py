"""Weighted NRS attribute reduction (Hu et al., WNRS).

Attribute weights derive from per-attribute neighborhood entropy; forward
greedy selection uses standard Hu NRS dependency with weighted significance
sig_w(a|B) = w_a * (gamma(B union {a}) - gamma(B)).
"""

from __future__ import annotations

import numpy as np

from ccar.nrs import (
    attribute_significance,
    block_distance,
    dependency_degree,
    normalize_features,
    row_blocks,
)


def _attribute_entropy(
    X: np.ndarray,
    y: np.ndarray,
    attr: int,
    delta: float,
) -> float:
    """Mean decision entropy in delta-neighborhoods defined by attribute a alone."""
    n = X.shape[0]
    idx = np.array([attr], dtype=int)
    classes = np.unique(y)
    ent_sum = 0.0
    for s, e in row_blocks(n):
        nb = block_distance(X, s, e, idx) <= delta
        counts = np.stack([(nb & (y == c)[None, :]).sum(axis=1) for c in classes], axis=1)
        for row in counts:
            row = row[row > 0]
            p = row / row.sum()
            ent_sum += -float(np.sum(p * np.log(p + 1e-12)))
    return ent_sum / max(n, 1)


def attribute_weights(
    X: np.ndarray,
    y: np.ndarray,
    delta: float,
) -> np.ndarray:
    p = X.shape[1]
    ent = np.array([_attribute_entropy(X, y, a, delta) for a in range(p)], dtype=float)
    if ent.sum() <= 1e-12:
        return np.ones(p) / max(p, 1)
    return ent / ent.sum()


def weighted_significance(
    X: np.ndarray,
    y: np.ndarray,
    base_idx: np.ndarray,
    attr: int,
    weights: np.ndarray,
    delta: float,
) -> float:
    sig = attribute_significance(X, y, base_idx, attr, delta)
    return weights[attr] * sig


def wnrs_reduction(
    X: np.ndarray,
    y: np.ndarray,
    delta: float = 0.1,
    *,
    normalize: bool = True,
) -> list[int]:
    """Forward greedy WNRS reduct under standard gamma_delta; never empty."""
    X_work = normalize_features(X) if normalize else X.astype(float)
    y = np.asarray(y)
    n_attr = X_work.shape[1]
    if n_attr == 0:
        return []

    weights = attribute_weights(X_work, y, delta)
    all_idx = np.arange(n_attr, dtype=int)
    gamma_c = dependency_degree(X_work, y, all_idx, delta)
    selected: list[int] = []
    sel_idx = np.array([], dtype=int)

    while dependency_degree(X_work, y, sel_idx, delta) < gamma_c - 1e-12:
        remaining = [a for a in range(n_attr) if a not in selected]
        if not remaining:
            break
        best_a = max(
            remaining,
            key=lambda a: weighted_significance(
                X_work, y, sel_idx, a, weights, delta
            ),
        )
        sig = weighted_significance(X_work, y, sel_idx, best_a, weights, delta)
        if sig <= 1e-12:
            best_a = max(
                remaining,
                key=lambda a: attribute_significance(X_work, y, sel_idx, a, delta),
            )
            if attribute_significance(X_work, y, sel_idx, best_a, delta) <= 1e-12:
                break
        selected.append(best_a)
        sel_idx = np.array(selected, dtype=int)

    if not selected:
        selected = [int(np.argmax(weights))]
    return sorted(selected)
