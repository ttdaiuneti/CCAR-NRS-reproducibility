"""
GBNRS: Granular Ball Neighborhood Rough Set
Xia et al. 2020, IEEE TKDE
"GBNRS: A Novel Rough Set Algorithm for Fast 
 Adaptive Attribute Reduction in Classification"
"""
import numpy as np
from typing import List, Dict, Tuple, Set
from dataclasses import dataclass


@dataclass
class GranularBall:
    """Một granular ball = hypersphere bao một nhóm objects"""
    indices: List[int]      # indices của objects trong ball
    center: np.ndarray      # centroid
    radius: float           # max distance từ center đến member
    purity: float           # tỷ lệ label phổ biến nhất
    dominant_label: int     # nhãn phổ biến nhất


def compute_ball_stats(X: np.ndarray,
                       labels: np.ndarray,
                       indices: List[int],
                       attrs: List[int]) -> GranularBall:
    """Tính các thống kê của một granular ball."""
    pts = X[np.ix_(indices, attrs)]
    center = pts.mean(axis=0)

    # Radius = max distance từ center
    dists = np.sqrt(((pts - center) ** 2).sum(axis=1))
    radius = float(dists.max()) if len(indices) > 1 else 0.0

    # Purity
    lbls = labels[indices]
    unique, counts = np.unique(lbls, return_counts=True)
    dominant_label = int(unique[counts.argmax()])
    purity = float(counts.max()) / len(indices)

    return GranularBall(
        indices=indices,
        center=center,
        radius=radius,
        purity=purity,
        dominant_label=dominant_label
    )


def generate_granular_balls(X: np.ndarray,
                            labels: np.ndarray,
                            attrs: List[int],
                            min_sample: int = 2,
                            purity_threshold: float = 1.0,
                            max_depth: int = 20
                            ) -> List[GranularBall]:
    """
    Algorithm 1 (Xia 2020): Granular Ball Generation
    Split đệ quy cho đến khi purity >= threshold
    hoặc ball chỉ còn min_sample objects.
    """
    balls = []

    def split_ball(indices: List[int], depth: int = 0):
        if len(indices) < min_sample or depth >= max_depth:
            gb = compute_ball_stats(X, labels, indices, attrs)
            balls.append(gb)
            return

        gb = compute_ball_stats(X, labels, indices, attrs)

        if gb.purity >= purity_threshold or len(indices) == 1:
            balls.append(gb)
            return

        # Split theo K-means với K=2
        from sklearn.cluster import KMeans
        pts = X[np.ix_(indices, attrs)]

        if len(indices) == 2:
            balls.append(gb)
            return

        km = KMeans(n_clusters=2, n_init=3, random_state=42)
        cluster_labels = km.fit_predict(pts)

        group0 = [indices[i] for i in range(len(indices))
                  if cluster_labels[i] == 0]
        group1 = [indices[i] for i in range(len(indices))
                  if cluster_labels[i] == 1]

        if len(group0) >= 1:
            split_ball(group0, depth + 1)
        if len(group1) >= 1:
            split_ball(group1, depth + 1)

    split_ball(list(range(len(X))))
    return balls


def compute_gb_dependency(X: np.ndarray,
                          labels: np.ndarray,
                          attrs: List[int],
                          purity_threshold: float = 1.0
                          ) -> Tuple[float, List[GranularBall]]:
    """
    γ_{GB}(B, D) = số objects trong pure balls / |U|
    """
    balls = generate_granular_balls(
        X, labels, attrs,
        purity_threshold=purity_threshold
    )

    pos_count = sum(
        len(b.indices)
        for b in balls
        if b.purity >= purity_threshold
    )

    gamma = pos_count / len(X)
    return gamma, balls


def gbnrs_reduction(X: np.ndarray,
                    condition_attrs: List[int],
                    labels: np.ndarray,
                    purity_threshold: float = 1.0,
                    verbose: bool = False) -> Dict:
    """
    Algorithm 2 (Xia 2020): GBNRS Attribute Reduction
    Forward greedy dựa trên GB-dependency.
    """
    import time
    t_start = time.time()
    C = list(condition_attrs)

    # γ(C, D) với toàn bộ attributes
    gamma_C, _ = compute_gb_dependency(X, labels, C, purity_threshold)

    if verbose:
        print(f"[GBNRS] γ(C,D) = {gamma_C:.4f}")

    reduct = []

    while True:
        # Kiểm tra đạt đủ dependency
        if reduct:
            g, _ = compute_gb_dependency(
                X, labels, reduct, purity_threshold)
            if abs(g - gamma_C) < 1e-10:
                break

        # Chọn attr tốt nhất
        best_attr, best_gamma = None, -1.0
        for a in C:
            if a in reduct:
                continue
            candidate = reduct + [a]
            g, _ = compute_gb_dependency(
                X, labels, candidate, purity_threshold)
            if g > best_gamma:
                best_gamma = g
                best_attr = a

        if best_attr is None:
            break
        reduct.append(best_attr)

        if verbose:
            print(f"  Add attr {best_attr}, γ = {best_gamma:.4f}")

    return {
        "reduct": reduct,
        "gamma_C": gamma_C,
        "gamma_reduct": compute_gb_dependency(
            X, labels, reduct, purity_threshold)[0],
        "time_total": time.time() - t_start,
    }
