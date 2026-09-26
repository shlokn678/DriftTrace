"""Per-node drift engine (FR-9).

Runs KS (continuous nodes) and PSI (continuous + categorical) for every drift-checkable
declared node, against the versioned training baseline, then combines the per-test
verdicts into a single per-node verdict. Raw KS and PSI evidence is retained; no hidden
scoring is introduced.

Combine rule (documented):
- If both tests are INSUFFICIENT_DATA (or the only applicable test is) -> INSUFFICIENT_DATA.
- Otherwise the node verdict is the most severe available verdict, ordered
  DRIFT > WARNING > STABLE (INSUFFICIENT_DATA from one test is ignored if the other
  test produced a usable verdict).

Deterministic and broker-free (NFR-8).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import pandas as pd

from drifttrace.drift.baseline import Baseline
from drifttrace.drift.config import DriftConfig
from drifttrace.drift.ks import KSResult, ks_test
from drifttrace.drift.psi import PSIResult, psi_categorical, psi_continuous
from drifttrace.drift.verdict import Verdict


@dataclass
class NodeDriftResult:
    """Combined per-node drift result with retained KS/PSI evidence."""

    node: str
    verdict: str
    ks: dict | None = None
    psi: dict | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DriftReport:
    """Drift results for all checked nodes in one window."""

    window_id: str
    baseline_version: str
    nodes: dict[str, NodeDriftResult] = field(default_factory=dict)

    @property
    def drifted_nodes(self) -> list[str]:
        return [n for n, r in self.nodes.items() if r.verdict == str(Verdict.DRIFT)]

    def to_dict(self) -> dict:
        return {
            "window_id": self.window_id,
            "baseline_version": self.baseline_version,
            "nodes": {n: r.to_dict() for n, r in self.nodes.items()},
            "drifted_nodes": self.drifted_nodes,
        }


def _combine(ks: KSResult | None, psi: PSIResult | None) -> str:
    """Combine KS and PSI into one node verdict (documented, no hidden scoring).

    KS is a hypothesis test that becomes over-sensitive at large sample sizes (it will
    reject on trivial, practically-meaningless differences when the baseline is large).
    PSI is an effect-size measure that is robust to sample size. To avoid false DRIFT
    from KS over-sensitivity while still using both signals (FR-9.1), the rule is:

    - DRIFT  if PSI >= psi_drift (strong effect), OR KS says DRIFT AND PSI >= psi_warning
             (KS-significant AND a non-trivial effect size).
    - WARNING if PSI is in the warning band, OR KS says DRIFT but PSI is below warning
             (statistically flagged but small effect -> worth watching, not alerting).
    - STABLE  otherwise.
    - INSUFFICIENT_DATA if neither test produced a usable verdict.
    """
    ks_usable = ks is not None and ks.verdict != str(Verdict.INSUFFICIENT_DATA)
    psi_usable = psi is not None and psi.verdict != str(Verdict.INSUFFICIENT_DATA)
    if not ks_usable and not psi_usable:
        return str(Verdict.INSUFFICIENT_DATA)

    ks_drift = ks_usable and ks.verdict == str(Verdict.DRIFT)  # type: ignore[union-attr]
    psi_verdict = psi.verdict if psi_usable else str(Verdict.STABLE)  # type: ignore[union-attr]

    if psi_verdict == str(Verdict.DRIFT):
        return str(Verdict.DRIFT)
    if ks_drift and psi_verdict == str(Verdict.WARNING):
        return str(Verdict.DRIFT)
    if psi_verdict == str(Verdict.WARNING):
        return str(Verdict.WARNING)
    if ks_drift:
        # KS-significant but PSI shows a small/negligible effect -> watch, don't alert.
        return str(Verdict.WARNING)
    return str(Verdict.STABLE)


def detect_drift(
    window_values: dict[str, list[float]],
    window_categoricals: dict[str, list[str]],
    baseline: Baseline,
    config: DriftConfig,
    *,
    window_id: str,
) -> DriftReport:
    """Run KS+PSI per node over one window's feature values against the baseline.

    ``window_values`` maps continuous node -> list of values; ``window_categoricals``
    maps categorical node -> list of category labels. Only nodes present in the
    baseline are checked.
    """
    report = DriftReport(window_id=window_id, baseline_version=baseline.model_version)

    for node, nb in baseline.nodes.items():
        thr = config.thresholds_for(node)
        ks_res: KSResult | None = None
        psi_res: PSIResult | None = None

        if nb.kind == "continuous":
            current = window_values.get(node, [])
            ks_res = ks_test(
                node,
                current,
                nb.values,
                window_id=window_id,
                baseline_version=baseline.model_version,
                threshold=thr.ks_pvalue,
                min_samples=config.min_samples,
            )
            psi_res = psi_continuous(
                node,
                current,
                nb.values,
                window_id=window_id,
                baseline_version=baseline.model_version,
                n_bins=config.psi_bins,
                psi_warning=thr.psi_warning,
                psi_drift=thr.psi_drift,
                min_samples=config.min_samples,
            )
        else:  # categorical
            current_cats = window_categoricals.get(node, [])
            psi_res = psi_categorical(
                node,
                current_cats,
                nb.categories,
                window_id=window_id,
                baseline_version=baseline.model_version,
                baseline_n=nb.count,
                psi_warning=thr.psi_warning,
                psi_drift=thr.psi_drift,
                min_samples=config.min_samples,
            )

        report.nodes[node] = NodeDriftResult(
            node=node,
            verdict=_combine(ks_res, psi_res),
            ks=ks_res.to_dict() if ks_res is not None else None,
            psi=psi_res.to_dict() if psi_res is not None else None,
        )

    return report


def window_feature_arrays(
    events_features: list[dict[str, float]],
    continuous_nodes: list[str],
) -> dict[str, list[float]]:
    """Collect per-node value lists from a window's event feature dicts."""
    out: dict[str, list[float]] = {n: [] for n in continuous_nodes}
    for feats in events_features:
        for node in continuous_nodes:
            if node in feats and feats[node] is not None:
                out[node].append(float(feats[node]))
    return out


def drift_report_from_frame(
    frame: pd.DataFrame,
    baseline: Baseline,
    config: DriftConfig,
    *,
    window_id: str,
) -> DriftReport:
    """Convenience for batch/testing: build a DriftReport from a DataFrame window."""
    values: dict[str, list[float]] = {}
    cats: dict[str, list[str]] = {}
    for node, nb in baseline.nodes.items():
        if node not in frame.columns:
            continue
        if nb.kind == "continuous":
            values[node] = [float(x) for x in frame[node].dropna().tolist()]
        else:
            cats[node] = [str(x) for x in frame[node].dropna().tolist()]
    return detect_drift(values, cats, baseline, config, window_id=window_id)
