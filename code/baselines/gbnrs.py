"""GBNRS attribute reduction (Xia et al., IEEE TKDE 2020).

Using granular balls with purity_threshold=0.9 for robust performance.
"""

from __future__ import annotations

import numpy as np
from typing import List, Tuple
from dataclasses import dataclass


@dataclass
class GranularBall:
    """A granular ball = hypersphere enclosing a group of objects."""
    indices: List[int]
    center: np.ndarray
    radius: float
    purity: float
    dominant_label: int


def compute_ball_stats(X: np.ndarray, labels: np.ndarray, indices: List[int], attrs: List[int]) -> GranularBall:
    """Compute statistics of a granular ball."""
    pts = X[np.ix_(indices, attrs)]
    center = pts.mean(axis=0)
    dists = np.sqrt(((pts - center) ** 2).sum(axis=1))
    radius = float(dists.max()) if len(indices) > 1 else 0.0
    lbls = labels[indices]
    unique, counts = np.unique(lbls, return_counts=True)
    dominant_label = int(unique[counts.argmax()])
    purity = float(counts.max()) / len(indices)
    return GranularBall(indices, center, radius, purity, dominant_label)


def generate_granular_balls(X: np.ndarray, labels: np.ndarray, attrs: List[int], 
                           min_sample: int = 2, purity_threshold: float = 0.9, max_depth: int = 20) -> List[GranularBall]:
    """Algorithm 1 (Xia 2020): Granular Ball Generation."""
    balls = []

    def split_ball(indices: List[int], depth: int = 0):
        if len(indices) < min_sample or depth >= max_depth:
            balls.append(compute_ball_stats(X, labels, indices, attrs))
            return
        gb = compute_ball_stats(X, labels, indices, attrs)
        if gb.purity >= purity_threshold or len(indices) == 1:
            balls.append(gb)
            return
        from sklearn.cluster import KMeans
        pts = X[np.ix_(indices, attrs)]
        if len(indices) == 2:
            balls.append(gb)
            return
        km = KMeans(n_clusters=2, n_init=3, random_state=42)
        cluster_labels = km.fit_predict(pts)
        group0 = [indices[i] for i in range(len(indices)) if cluster_labels[i] == 0]
        group1 = [indices[i] for i in range(len(indices)) if cluster_labels[i] == 1]
        if len(group0) >= 1:
            split_ball(group0, depth + 1)
        if len(group1) >= 1:
            split_ball(group1, depth + 1)

    split_ball(list(range(len(X))))
    return balls


def compute_gb_dependency(X: np.ndarray, labels: np.ndarray, attrs: List[int], 
                          purity_threshold: float = 0.9) -> Tuple[float, List[GranularBall]]:
    """γ_{GB}(B, D) = number of objects in pure balls / |U|."""
    balls = generate_granular_balls(X, labels, attrs, purity_threshold=purity_threshold)
    pos_count = sum(len(b.indices) for b in balls if b.purity >= purity_threshold)
    return pos_count / len(X), balls


def gbnrs_reduction(X: np.ndarray, y: np.ndarray, delta: float = 0.1, *, 
                    purity_threshold: float = 0.9, verbose: bool = False) -> list[int]:
    """Algorithm 2 (Xia 2020): GBNRS Attribute Reduction."""
    import time
    t_start = time.time()
    del delta  # GBNRS uses adaptive granular balls, not delta-neighborhoods
    # Feature scaling harmonized with the other methods (MinMax to [0,1]) so that
    # all algorithms see identically preprocessed inputs. GBNRS's *criterion*
    # remains granular-ball purity by design -- unlike the neighborhood-based
    # methods it does not optimize gamma_delta, which is why its reducts need not
    # satisfy the gamma_delta validity test. That is a paradigm difference, not a
    # defect, and is reported as such.
    from ccar.nrs import normalize_features

    X_norm = normalize_features(X)
    C = list(range(X.shape[1]))
    gamma_C, _ = compute_gb_dependency(X_norm, y, C, purity_threshold)
    if verbose:
        print(f"[GBNRS] γ(C,D) = {gamma_C:.4f}")
    reduct = []
    while True:
        if reduct:
            g, _ = compute_gb_dependency(X_norm, y, reduct, purity_threshold)
            if abs(g - gamma_C) < 1e-10:
                break
        best_attr, best_gamma = None, -1.0
        for a in C:
            if a in reduct:
                continue
            candidate = reduct + [a]
            g, _ = compute_gb_dependency(X_norm, y, candidate, purity_threshold)
            if g > best_gamma:
                best_gamma = g
                best_attr = a
        if best_attr is None:
            break
        reduct.append(best_attr)
        if verbose:
            print(f"  Add attr {best_attr}, γ = {best_gamma:.4f}")
    if verbose:
        print(f"Time: {time.time() - t_start:.4f}s")
    return reduct
