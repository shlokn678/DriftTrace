"""Kolmogorov-Smirnov drift detector (FR-9.1, FR-9.4).

Compares a current monitoring-window sample against the versioned training baseline
sample for a continuous node using the two-sample KS test (SciPy). Produces
structured, deterministic evidence and respects the minimum sample-size rule
(FR-9.5): windows below ``min_samples`` return INSUFFICIENT_DATA rather than a verdict.

Verdict rule (FR-9.4): KS p-value < ks_pvalue threshold -> DRIFT, else STABLE.
(The KS test alone does not define a WARNING band; WARNING comes from PSI.)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy import stats

from drifttrace.drift.verdict import Verdict


@dataclass
class KSResult:
    """Structured KS evidence for one node/window."""

    node: str
    window_id: str
    baseline_version: str
    current_n: int
    baseline_n: int
    statistic: float | None
    p_value: float | None
    threshold: float
    verdict: str

    def to_dict(self) -> dict:
        return asdict(self)


def ks_test(
    node: str,
    current: np.ndarray | list[float],
    baseline: np.ndarray | list[float],
    *,
    window_id: str,
    baseline_version: str,
    threshold: float = 0.05,
    min_samples: int = 30,
) -> KSResult:
    """Run the two-sample KS test of ``current`` vs ``baseline`` for a continuous node.

    Returns a :class:`KSResult`. If either sample has fewer than ``min_samples``
    observations, the verdict is INSUFFICIENT_DATA and statistic/p-value are None.
    """
    cur = np.asarray(current, dtype=float)
    base = np.asarray(baseline, dtype=float)
    cur = cur[~np.isnan(cur)]
    base = base[~np.isnan(base)]

    if cur.size < min_samples or base.size < min_samples:
        return KSResult(
            node=node,
            window_id=window_id,
            baseline_version=baseline_version,
            current_n=int(cur.size),
            baseline_n=int(base.size),
            statistic=None,
            p_value=None,
            threshold=threshold,
            verdict=str(Verdict.INSUFFICIENT_DATA),
        )

    result = stats.ks_2samp(cur, base)
    statistic = float(result.statistic)
    p_value = float(result.pvalue)
    verdict = Verdict.DRIFT if p_value < threshold else Verdict.STABLE

    return KSResult(
        node=node,
        window_id=window_id,
        baseline_version=baseline_version,
        current_n=int(cur.size),
        baseline_n=int(base.size),
        statistic=statistic,
        p_value=p_value,
        threshold=threshold,
        verdict=str(verdict),
    )
