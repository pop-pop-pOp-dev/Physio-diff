from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import numpy as np
from scipy.stats import wilcoxon


def values_from_seed_dict(vals: Dict[str, float]) -> np.ndarray:
    return np.asarray(list(vals.values()), dtype=np.float64)


def median_iqr(vals: Dict[str, float]) -> Tuple[float, float, float]:
    arr = values_from_seed_dict(vals)
    if arr.size == 0:
        return 0.0, 0.0, 0.0
    q1, med, q3 = np.percentile(arr, [25, 50, 75])
    return float(med), float(q1), float(q3)


def bootstrap_ci(vals: Dict[str, float], n_boot: int = 2000, ci: float = 0.95) -> Tuple[float, float]:
    arr = values_from_seed_dict(vals)
    if arr.size == 0:
        return 0.0, 0.0
    if arr.size == 1:
        return float(arr[0]), float(arr[0])
    rng = np.random.default_rng(1234)
    stats = []
    for _ in range(int(n_boot)):
        sample = rng.choice(arr, size=arr.size, replace=True)
        stats.append(float(np.median(sample)))
    lo = (1.0 - float(ci)) / 2.0
    hi = 1.0 - lo
    return float(np.quantile(stats, lo)), float(np.quantile(stats, hi))


def paired_wilcoxon(base: Dict[str, float], other: Dict[str, float]) -> float | None:
    common = sorted(set(base.keys()) & set(other.keys()))
    if len(common) < 2:
        return None
    b = np.asarray([base[k] for k in common], dtype=np.float64)
    o = np.asarray([other[k] for k in common], dtype=np.float64)
    try:
        _, p = wilcoxon(b, o, alternative="two-sided")
        return float(p)
    except Exception:
        return None


def rank_biserial(base: Dict[str, float], other: Dict[str, float]) -> float | None:
    common = sorted(set(base.keys()) & set(other.keys()))
    if len(common) < 2:
        return None
    diff = np.asarray([base[k] - other[k] for k in common], dtype=np.float64)
    diff = diff[diff != 0]
    if diff.size == 0:
        return 0.0
    ranks = np.argsort(np.argsort(np.abs(diff))) + 1
    w_pos = float(ranks[diff > 0].sum())
    w_neg = float(ranks[diff < 0].sum())
    denom = diff.size * (diff.size + 1) / 2.0
    return float((w_pos - w_neg) / denom)


def benjamini_hochberg(p_values: Iterable[float | None]) -> List[float | None]:
    indexed = [(idx, p) for idx, p in enumerate(p_values) if p is not None]
    if not indexed:
        return [None for _ in p_values]
    m = len(indexed)
    indexed.sort(key=lambda item: item[1])
    adjusted = [None for _ in p_values]
    running = 1.0
    for rank, (idx, p) in enumerate(reversed(indexed), start=1):
        bh = min(running, p * m / (m - rank + 1))
        running = bh
        adjusted[idx] = float(min(1.0, bh))
    return adjusted
