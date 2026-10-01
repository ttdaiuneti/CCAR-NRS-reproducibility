"""Result types for CCAR algorithms."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CCARResult:
    reduct_indices: list[int]
    core_indices: list[int]
    gamma_c: float
    n_pairs: int
    n_family: int
    used_surrogate: bool
