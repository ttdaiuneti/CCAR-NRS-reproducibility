"""Shared CCAR pipeline and the surrogate seed used when CORE is empty."""

from __future__ import annotations

import numpy as np

from ccar.conflicting import conflicting_pairs, m_res_family
from ccar.core import compute_core_pos
from ccar.hitting_set import greedy_hitting_set, minimal_hitting_set_exact
from ccar.nrs import attribute_significance, backward_prune, dependency_degree, normalize_features
from ccar.result import CCARResult


def _surrogate_core(
    X: np.ndarray,
    y: np.ndarray,
    delta: float,
    tau: int,
) -> list[int]:
    """Top-tau attributes by sig(a | empty set) (Definition 3.12)."""
    n_attr = X.shape[1]
    scores = [
        (attribute_significance(X, y, np.array([], dtype=int), a, delta), a)
        for a in range(n_attr)
    ]
    scores.sort(reverse=True)
    tau = max(1, min(tau, n_attr))
    return [a for _, a in scores[:tau]]


def ccar_pipeline(
    X_work: np.ndarray,
    y: np.ndarray,
    delta: float,
    *,
    exact_hitting: bool,
    tau: int = 1,
) -> CCARResult:
    """CORE -> P_delta (restricted to M_POS) -> F -> hitting set -> R.

    CCAR-E (exact_hitting) with a non-empty core returns CORE u S_min with S_min
    a minimum hitting set of F, which is a minimum-cardinality (approximate)
    reduct by Theorem 3.7 / Lemma A.1; no pruning is needed. The two paths
    without that guarantee are pruned backward so that every output is a
    minimal (approximate) reduct: the greedy cover of CCAR-H (attributes tried
    in reverse selection order) and the surrogate path (S, then the seed).
    Core attributes are never tried, since removing one lowers gamma_delta.
    """
    y = np.asarray(y)
    n_attr = X_work.shape[1]
    all_idx = np.arange(n_attr)

    core, gamma_c, pos = compute_core_pos(X_work, y, delta)
    used_surrogate = not core
    seed = list(core) if core else _surrogate_core(X_work, y, delta, tau)
    seed_idx = np.array(seed, dtype=int)

    pairs = conflicting_pairs(X_work, y, seed_idx, delta, pos_mask=pos)
    family, weights = m_res_family(X_work, pairs, seed_idx, all_idx, delta)
    non_seed = [int(a) for a in np.setdiff1d(all_idx, seed_idx)]
    if not family:
        s: list[int] = []
    elif exact_hitting:
        s = minimal_hitting_set_exact(non_seed, family, weights)
    else:
        tie = {a: attribute_significance(X_work, y, seed_idx, a, delta) for a in non_seed}
        s = greedy_hitting_set(non_seed, family, tie_scores=tie, weights=weights)

    reduct = sorted(set(seed) | set(s))
    if used_surrogate:
        removable = (sorted(s) if exact_hitting else list(reversed(s))) + seed
        reduct = backward_prune(X_work, y, reduct, removable, gamma_c, delta)
    elif not exact_hitting:
        reduct = backward_prune(X_work, y, reduct, list(reversed(s)), gamma_c, delta)
    return CCARResult(
        reduct_indices=reduct,
        core_indices=list(core),
        gamma_c=gamma_c,
        n_pairs=int(len(pairs)),
        n_family=int(sum(weights)) if weights else 0,
        used_surrogate=used_surrogate,
    )


def ccar_nrs_surrogate(
    X: np.ndarray,
    y: np.ndarray,
    delta: float = 0.1,
    tau: int = 1,
    *,
    exact_hitting: bool = True,
    normalize: bool = True,
) -> CCARResult:
    """CCAR with the surrogate seed when CORE is empty (identical to CCAR-E/H otherwise)."""
    X_work = normalize_features(X) if normalize else X.astype(float)
    return ccar_pipeline(X_work, y, delta, exact_hitting=exact_hitting, tau=tau)


def verify_minimal(
    X_work: np.ndarray,
    y: np.ndarray,
    reduct: list[int],
    gamma_c: float,
    delta: float,
) -> bool:
    """True iff no single attribute can be dropped from `reduct` keeping gamma >= gamma_c
    (by monotonicity this is equivalent to minimality)."""
    for a in reduct:
        trial = np.array([b for b in reduct if b != a], dtype=int)
        if trial.size and dependency_degree(X_work, y, trial, delta) >= gamma_c - 1e-12:
            return False
    return True
