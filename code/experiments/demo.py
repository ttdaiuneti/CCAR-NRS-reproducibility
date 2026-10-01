"""Quick demo on sklearn UCI dataset."""

from __future__ import annotations

import argparse
import time

import numpy as np
from sklearn.datasets import load_wine
from sklearn.preprocessing import LabelEncoder

from ccar.ccar_exact import ccar_nrs_exact, verify_reduct
from ccar.ccar_heuristic import ccar_nrs_heuristic
from ccar.ccar_surrogate import ccar_nrs_surrogate

try:
    from baselines.greedy_core_init import core_init_greedy
except ImportError:
    from baselines.greedy_core_init import core_init_greedy  # type: ignore


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="wine")
    parser.add_argument("--delta", type=float, default=0.1)
    args = parser.parse_args()

    if args.dataset == "wine":
        data = load_wine()
        X, y = data.data, data.target
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    y = LabelEncoder().fit_transform(y)
    print(f"Dataset: {args.dataset}, |U|={X.shape[0]}, |C|={X.shape[1]}, delta={args.delta}")

    for name, fn in [
        ("CCAR-Exact", lambda: ccar_nrs_exact(X, y, args.delta)),
        ("CCAR-H", lambda: ccar_nrs_heuristic(X, y, args.delta)),
        ("CCAR-Surrogate", lambda: ccar_nrs_surrogate(X, y, args.delta)),
    ]:
        t0 = time.perf_counter()
        res = fn()
        dt = time.perf_counter() - t0
        ok = verify_reduct(X, y, res.reduct_indices, res.gamma_c, args.delta)
        print(
            f"{name}: |R|={len(res.reduct_indices)} |CORE|={len(res.core_indices)} "
            f"|P|={res.n_pairs} gamma={res.gamma_c:.4f} valid={ok} time={dt:.3f}s"
        )

    t0 = time.perf_counter()
    r_g = core_init_greedy(X, y, args.delta)
    print(f"Core-init-Greedy: |R|={len(r_g)} time={time.perf_counter()-t0:.3f}s")


if __name__ == "__main__":
    main()
