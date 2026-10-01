"""NIP-NRS: neighbor-inconsistent-pair greedy reduction for neighborhood rough sets.

An adaptation, not a reimplementation, of the significance measure of
Dai, Hu, Hu and Huang (2018), "Neighbor inconsistent pair selection for
attribute reduction by rough set approach", IEEE Trans. Fuzzy Syst. 26(2),
937-950 (NICP; algorithms QNIPS/QNIPS2).

The published algorithms work on Pawlak (equivalence-relation) decision
tables: objects are sorted by the partition U/R, a "neighbor pair" is two
*adjacent* objects in that order, and sigNIP(R, a, D) counts the adjacent
inconsistent pairs inside equivalence classes that attribute a separates.
A delta-neighborhood relation is not transitive and induces no partition or
sort order, so QNIPS has no direct counterpart under the NRS criterion used
by every other method in this benchmark. We therefore keep the NICP
significance idea on the NRS pair space:

    NIP_B = { (i, j) : Delta_B(x_i, x_j) <= delta  and  d(x_i) != d(x_j) },

select forward the attribute that leaves the fewest pairs in NIP_B, stop when
|NIP_B| reaches |NIP_C|, and drop attributes whose removal does not increase
|NIP_B| (the analogue of QNIPS's redundancy test, Algorithm 5). The pair
counts use the shared blocked kernel of ccar.nrs.
"""

from __future__ import annotations

import numpy as np

from ccar.nrs import block_distance, normalize_features, row_blocks


def neighbor_inconsistent_pairs(
    X: np.ndarray,
    y: np.ndarray,
    attr_idx: np.ndarray,
    delta: float,
) -> int:
    """Count ordered pairs in NIP_B (X already normalized; max-metric)."""
    n = X.shape[0]
    if n == 0:
        return 0
    y = np.asarray(y)
    attr_idx = np.asarray(attr_idx, dtype=int)
    if attr_idx.size == 0:
        return int(np.sum(y[:, None] != y[None, :]))
    count = 0
    for s, e in row_blocks(n):
        nb = block_distance(X, s, e, attr_idx) <= delta
        count += int(np.sum(nb & (y[s:e, None] != y[None, :])))
    return count


def nicp_reduction(
    X: np.ndarray,
    y: np.ndarray,
    delta: float = 0.1,
    *,
    normalize: bool = True,
    verbose: bool = False,
) -> list[int]:
    """NIP-NRS: forward greedy on |NIP_B|, then backward elimination.

    Returns the selected attribute indices (relative to the original columns).
    """
    X_work = normalize_features(X) if normalize else X.astype(float)
    n_attrs = X_work.shape[1]
    C = np.arange(n_attrs)

    target_nip = neighbor_inconsistent_pairs(X_work, y, C, delta)
    if verbose:
        print(f"[NIP-NRS] |NIP_C| = {target_nip}")

    selected: list[int] = []
    remaining = set(range(n_attrs))
    cur_nip = neighbor_inconsistent_pairs(X_work, y, np.array([], dtype=int), delta)

    while remaining and cur_nip > target_nip:
        best_attr, best_nip = None, None
        for a in sorted(remaining):
            nip = neighbor_inconsistent_pairs(X_work, y, np.array(selected + [a], dtype=int), delta)
            if best_nip is None or nip < best_nip:
                best_nip, best_attr = nip, a
        if best_attr is None or best_nip >= cur_nip:
            # No attribute removes any further inconsistent pair.
            break
        selected.append(best_attr)
        remaining.discard(best_attr)
        cur_nip = best_nip
        if verbose:
            print(f"  + attr {best_attr}: |NIP| -> {best_nip}")

    # Backward elimination: drop attributes whose removal does not increase |NIP|.
    changed = True
    while changed and len(selected) > 1:
        changed = False
        for a in list(selected):
            trial = [b for b in selected if b != a]
            nip_trial = neighbor_inconsistent_pairs(X_work, y, np.array(trial, dtype=int), delta)
            if nip_trial <= cur_nip:
                selected = trial
                cur_nip = nip_trial
                changed = True
                if verbose:
                    print(f"  - attr {a} (redundant)")
                break

    return sorted(selected)


nip_nrs_reduction = nicp_reduction
