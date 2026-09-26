"""Population Stability Index (PSI) drift detector (FR-9.1, FR-9.4).

Compares a current window distribution against the versioned training baseline.

- Continuous nodes: bin the baseline into quantile bins (edges derived from the
  baseline), then compare current-vs-baseline proportions per bin.
- Categorical nodes: use the declared categories as bins.

Zero / near-zero proportions are handled by epsilon smoothing so PSI stays finite.

Verdict rule (FR-9.4): PSI < psi_warning -> STABLE; psi_warning <= PSI < psi_drift ->
WARNING; PSI >= psi_drift -> DRIFT. Below the minimum sample size -> INSUFFICIENT_DATA.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from drifttrace.drift.verdict import Verdict

_EPS = 1e-6


@dataclass
class PSIBin:
    """Per-bin PSI contribution (structured evidence)."""

    label: str
    baseline_prop: float
    current_prop: float
    contribution: float


@dataclass
class PSIResult:
    """Structured PSI evidence for one node/window."""

    node: str
    window_id: str
    baseline_version: str
    current_n: int
    baseline_n: int
    psi: float | None
    bins: list[PSIBin] = field(default_factory=list)
    psi_warning: float = 0.1
    psi_drift: float = 0.2
    verdict: str = str(Verdict.INSUFFICIENT_DATA)

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


def _verdict_for(psi: float, warning: float, drift: float) -> Verdict:
    if psi >= drift:
        return Verdict.DRIFT
    if psi >= warning:
        return Verdict.WARNING
    return Verdict.STABLE


def _quantile_edges(baseline: np.ndarray, n_bins: int) -> np.ndarray:
    """Quantile bin edges from the baseline, deduplicated, with open outer edges."""
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(baseline, quantiles)
    edges = np.unique(edges)
    # Ensure the outer edges capture values outside the baseline range.
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def _proportions(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    counts, _ = np.histogram(values, bins=edges)
    total = counts.sum()
    if total == 0:
        return np.zeros(len(counts), dtype=float)
    return counts / total


def psi_continuous(
    node: str,
    current: np.ndarray | list[float],
    baseline: np.ndarray | list[float],
    *,
    window_id: str,
    baseline_version: str,
    n_bins: int = 10,
    psi_warning: float = 0.1,
    psi_drift: float = 0.2,
    min_samples: int = 30,
) -> PSIResult:
    """Compute PSI for a continuous node using baseline-derived quantile bins."""
    cur = np.asarray(current, dtype=float)
    base = np.asarray(baseline, dtype=float)
    cur = cur[~np.isnan(cur)]
    base = base[~np.isnan(base)]

    if cur.size < min_samples or base.size < min_samples:
        return PSIResult(
            node=node,
            window_id=window_id,
            baseline_version=baseline_version,
            current_n=int(cur.size),
            baseline_n=int(base.size),
            psi=None,
            psi_warning=psi_warning,
            psi_drift=psi_drift,
            verdict=str(Verdict.INSUFFICIENT_DATA),
        )

    edges = _quantile_edges(base, n_bins)
    base_prop = _proportions(base, edges)
    cur_prop = _proportions(cur, edges)

    bins: list[PSIBin] = []
    psi_total = 0.0
    for i in range(len(base_prop)):
        b = max(float(base_prop[i]), _EPS)
        c = max(float(cur_prop[i]), _EPS)
        contribution = (c - b) * float(np.log(c / b))
        psi_total += contribution
        bins.append(
            PSIBin(
                label=f"[{edges[i]:.4g}, {edges[i + 1]:.4g})",
                baseline_prop=float(base_prop[i]),
                current_prop=float(cur_prop[i]),
                contribution=contribution,
            )
        )

    return PSIResult(
        node=node,
        window_id=window_id,
        baseline_version=baseline_version,
        current_n=int(cur.size),
        baseline_n=int(base.size),
        psi=float(psi_total),
        bins=bins,
        psi_warning=psi_warning,
        psi_drift=psi_drift,
        verdict=str(_verdict_for(psi_total, psi_warning, psi_drift)),
    )


def psi_categorical(
    node: str,
    current: list[str] | np.ndarray,
    baseline_props: dict[str, float],
    *,
    window_id: str,
    baseline_version: str,
    baseline_n: int,
    psi_warning: float = 0.1,
    psi_drift: float = 0.2,
    min_samples: int = 30,
) -> PSIResult:
    """Compute PSI for a categorical node against baseline category proportions."""
    cur = np.asarray(list(current), dtype=object)
    if cur.size < min_samples:
        return PSIResult(
            node=node,
            window_id=window_id,
            baseline_version=baseline_version,
            current_n=int(cur.size),
            baseline_n=int(baseline_n),
            psi=None,
            psi_warning=psi_warning,
            psi_drift=psi_drift,
            verdict=str(Verdict.INSUFFICIENT_DATA),
        )

    categories = list(baseline_props.keys())
    # Include any unseen categories present in the current window.
    for v in {str(x) for x in cur}:
        if v not in categories:
            categories.append(v)

    cur_counts = dict.fromkeys(categories, 0)
    for x in cur:
        cur_counts[str(x)] = cur_counts.get(str(x), 0) + 1
    cur_total = float(cur.size)

    bins: list[PSIBin] = []
    psi_total = 0.0
    for cat in categories:
        b = max(float(baseline_props.get(cat, 0.0)), _EPS)
        c = max(cur_counts.get(cat, 0) / cur_total, _EPS)
        contribution = (c - b) * float(np.log(c / b))
        psi_total += contribution
        bins.append(
            PSIBin(
                label=str(cat),
                baseline_prop=float(baseline_props.get(cat, 0.0)),
                current_prop=float(cur_counts.get(cat, 0) / cur_total),
                contribution=contribution,
            )
        )

    return PSIResult(
        node=node,
        window_id=window_id,
        baseline_version=baseline_version,
        current_n=int(cur.size),
        baseline_n=int(baseline_n),
        psi=float(psi_total),
        bins=bins,
        psi_warning=psi_warning,
        psi_drift=psi_drift,
        verdict=str(_verdict_for(psi_total, psi_warning, psi_drift)),
    )
