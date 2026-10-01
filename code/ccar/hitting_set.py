"""Minimal hitting set (exact B&B + greedy heuristic)."""

from __future__ import annotations


def _hits(set_a: frozenset[int], family: list[frozenset[int]]) -> bool:
    return all(bool(set_a & f) for f in family)


def minimal_hitting_set_exact(
    universe: list[int],
    family: list[frozenset[int]],
    weights: list[int] | None = None,
) -> list[int]:
    """Branch-and-bound minimum-cardinality hitting set on universe.

    `weights` are member multiplicities (see m_res_family); they only order the
    branching candidates, exactly as the duplicated members would."""
    if not family:
        return []
    w = dict(zip(family, weights)) if weights is not None else {f: 1 for f in family}
    best: list[int] | None = None

    def bound(size: int) -> bool:
        return best is not None and size >= len(best)

    def search(chosen: list[int], remaining: list[int], uncovered: list[frozenset[int]]):
        nonlocal best
        if not uncovered:
            if best is None or len(chosen) < len(best):
                best = list(chosen)
            return
        if bound(len(chosen)):
            return
        if not remaining:
            return
        # Pick set with fewest candidate hitters (MRV)
        f = min(uncovered, key=len)
        candidates = [a for a in remaining if a in f]
        if not candidates:
            return
        for a in sorted(candidates, key=lambda x: -sum(w[u] for u in uncovered if x in u)):
            new_uncovered = [u for u in uncovered if a not in u]
            new_rem = [x for x in remaining if x != a]
            search(chosen + [a], new_rem, new_uncovered)

    search([], list(universe), family)
    return best if best is not None else list(universe)


def greedy_hitting_set(
    universe: list[int],
    family: list[frozenset[int]],
    tie_scores: dict[int, float] | None = None,
    weights: list[int] | None = None,
) -> list[int]:
    """Greedy cover: max (multiplicity-weighted) frequency, tie-break by tie_scores.
    Returns attributes in selection order."""
    if not family:
        return []
    w = dict(zip(family, weights)) if weights is not None else {f: 1 for f in family}
    uncovered = list(family)
    chosen: list[int] = []
    available = set(universe)
    while uncovered:
        scores: dict[int, int] = {}
        for f in uncovered:
            for a in f:
                if a in available:
                    scores[a] = scores.get(a, 0) + w[f]
        if not scores:
            break
        max_count = max(scores.values())

        def key(a: int) -> tuple:
            ts = tie_scores.get(a, 0.0) if tie_scores else 0.0
            return (scores[a], ts, -a)

        candidates = [a for a, c in scores.items() if c == max_count]
        pick = max(candidates, key=key)
        chosen.append(pick)
        available.discard(pick)
        uncovered = [f for f in uncovered if pick not in f]
    return chosen
