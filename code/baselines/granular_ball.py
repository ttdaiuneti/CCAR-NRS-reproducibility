"""Granular-ball generation for GBNRS (Xia et al., TKDE 2020 / unified GBRS model).

Algorithm 1: recursive split (purity < 1 and |GB| > LBS) + heterogeneous overlap removal.
Ball radius: mean distance to center (Eq. 14 in Xia et al., arXiv:2201.03349).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from sklearn.cluster import KMeans
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)


@dataclass(frozen=True)
class GranularBall:
    indices: frozenset[int]
    center: tuple[float, ...]
    radius: float
    purity: float
    majority_label: int

    @property
    def size(self) -> int:
        return len(self.indices)


def _ball_stats(
    indices: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    attr_idx: np.ndarray,
) -> GranularBall:
    idx = np.unique(np.asarray(indices, dtype=int))
    if attr_idx.size:
        pts = X[idx][:, attr_idx]
    else:
        pts = np.zeros((idx.size, 0), dtype=float)

    if pts.size:
        center = pts.mean(axis=0)
        radius = float(np.mean(np.linalg.norm(pts - center, axis=1)))
    else:
        center = np.zeros(0, dtype=float)
        radius = 0.0

    labels = y[idx]
    counts = np.bincount(labels.astype(int))
    majority = int(np.argmax(counts))
    purity = float(counts[majority]) / idx.size
    return GranularBall(
        frozenset(int(i) for i in idx),
        tuple(float(v) for v in center),
        radius,
        purity,
        majority,
    )


def _split_ball_by_labels(
    ball: GranularBall,
    X: np.ndarray,
    y: np.ndarray,
    attr_idx: np.ndarray,
) -> list[GranularBall]:
    """Split ball into one sub-ball per decision class (fallback when k-means stalls)."""
    idx = np.array(sorted(ball.indices), dtype=int)
    labels = y[idx]
    out: list[GranularBall] = []
    for lab in np.unique(labels):
        sub = idx[labels == lab]
        if sub.size:
            out.append(_ball_stats(sub, X, y, attr_idx))
    return out if len(out) > 1 else [ball]


def _split_ball(
    ball: GranularBall,
    X: np.ndarray,
    y: np.ndarray,
    attr_idx: np.ndarray,
) -> list[GranularBall]:
    idx = np.array(sorted(ball.indices), dtype=int)
    labels = y[idx]
    unique = np.unique(labels)
    if unique.size <= 1 or attr_idx.size == 0:
        return [ball]

    pts = X[idx][:, attr_idx]
    inits = []
    for lab in unique:
        class_idx = idx[labels == lab]
        seed = int(class_idx[np.argmin(class_idx)])
        inits.append(X[seed, attr_idx])
    inits_arr = np.asarray(inits, dtype=float)

    k = int(unique.size)
    if idx.size <= k:
        return _split_ball_by_labels(ball, X, y, attr_idx)

    km = KMeans(
        n_clusters=k,
        init=inits_arr,
        n_init=1,
        max_iter=100,
        random_state=42,
    )
    assign = km.fit_predict(pts)

    out: list[GranularBall] = []
    for c in range(k):
        sub = idx[assign == c]
        if sub.size:
            out.append(_ball_stats(sub, X, y, attr_idx))

    if len(out) <= 1:
        return _split_ball_by_labels(ball, X, y, attr_idx)
    return out


def _center_vec(b: GranularBall) -> np.ndarray:
    return np.asarray(b.center, dtype=float)


def _balls_overlap(a: GranularBall, b: GranularBall) -> bool:
    if not a.center or not b.center:
        return a.indices & b.indices != frozenset()
    dist = float(np.linalg.norm(_center_vec(a) - _center_vec(b)))
    return dist < a.radius + b.radius - 1e-12


def _initial_granular_balls(
    X: np.ndarray,
    y: np.ndarray,
    attr_idx: np.ndarray,
    lbs: int,
) -> list[GranularBall]:
    n = X.shape[0]
    balls = [_ball_stats(np.arange(n, dtype=int), X, y, attr_idx)]

    for _ in range(n):
        nxt: list[GranularBall] = []
        for ball in balls:
            if ball.purity < 1.0 - 1e-12 and ball.size > lbs:
                nxt.extend(_split_ball(ball, X, y, attr_idx))
            else:
                nxt.append(ball)
        if len(nxt) == len(balls):
            return nxt
        balls = nxt
    return balls


def _remove_heterogeneous_overlap(
    balls: list[GranularBall],
    X: np.ndarray,
    y: np.ndarray,
    attr_idx: np.ndarray,
) -> list[GranularBall]:
    """Algorithm 1 (overlap phase): split larger ball when heterogeneous balls overlap."""
    balls = list(balls)
    max_rounds = max(100, len(balls) * 20)
    for _ in range(max_rounds):
        found = False
        for i in range(len(balls)):
            for j in range(i + 1, len(balls)):
                bi, bj = balls[i], balls[j]
                if bi.majority_label == bj.majority_label:
                    continue
                if not _balls_overlap(bi, bj):
                    continue
                li = i if bi.size >= bj.size else j
                before = balls[li]
                subs = _split_ball(before, X, y, attr_idx)
                if len(subs) == 1 and subs[0].size == before.size:
                    subs = _split_ball_by_labels(before, X, y, attr_idx)
                if len(subs) == 1 and subs[0].size == before.size:
                    continue
                balls = balls[:li] + subs + balls[li + 1 :]
                found = True
                break
            if found:
                break
        if not found:
            break
    return balls


def generate_granular_balls(
    X: np.ndarray,
    y: np.ndarray,
    attr_idx: np.ndarray,
    lbs: int,
) -> list[GranularBall]:
    """Algorithm 1 → NOLGBs."""
    balls = _initial_granular_balls(X, y, attr_idx, lbs)
    return _remove_heterogeneous_overlap(balls, X, y, attr_idx)


def gbnrs_dependency(
    X: np.ndarray,
    y: np.ndarray,
    attr_idx: np.ndarray,
    lbs: int,
) -> float:
    """γ_GBNRS(B,D) = |POS_B(D)| / |U| (Definition 15)."""
    n = X.shape[0]
    if n == 0:
        return 0.0
    idx = np.asarray(attr_idx, dtype=int)
    balls = generate_granular_balls(X, y, idx, lbs)
    pos = sum(b.size for b in balls if b.purity >= 0.9 - 1e-12)
    return pos / n


def select_lbs(X: np.ndarray, y: np.ndarray, n_attr: int) -> int:
    """Return fixed LBS=10 to prevent overfitting from grid-search."""
    # Fixed LBS to avoid overfitting with small LBS values
    return 10
