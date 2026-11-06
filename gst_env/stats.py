"""Uncertainty for small baseline runs. A pass rate over a handful of tasks
means little without an interval, so every reported rate carries one."""

import math
import random

Z95 = 1.959964


def wilson(passed: int, n: int, z: float = Z95) -> tuple[float, float]:
    """95% Wilson score interval for a pass rate. Stays inside [0, 1] and is
    honest at 0/n and n/n, where the normal approximation collapses to a point."""
    if n == 0:
        return 0.0, 1.0
    p = passed / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def bootstrap_mean(values: list[float], reps: int = 2000, seed: int = 0) -> tuple[float, float]:
    """95% percentile bootstrap interval for a mean reward."""
    if not values:
        return 0.0, 0.0
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(rng.choices(values, k=n)) / n for _ in range(reps))
    return means[int(0.025 * reps)], means[int(0.975 * reps) - 1]


def tasks_for_half_width(half_width: float, p: float = 0.5, z: float = Z95) -> int:
    """Tasks per cell so a pass-rate interval is at most +/- half_width (worst case p=0.5)."""
    return math.ceil(z * z * p * (1 - p) / (half_width * half_width))


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k from n samples of one task with c passes (Chen et al., 2021):
    the chance that at least one of k samples drawn without replacement passes."""
    if k > n:
        raise ValueError(f"need at least k={k} samples, got {n}")
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)
