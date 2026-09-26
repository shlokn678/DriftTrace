"""Drift detection configuration loader (FR-9.4/9.5).

Loads thresholds, per-node overrides, minimum sample size, and PSI binning from
``config/drift.yaml``. Pure: only yaml + stdlib, so detectors stay broker-free (NFR-8).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from drifttrace.config import get_paths


@dataclass(frozen=True)
class DriftThresholds:
    """Verdict thresholds (FR-9.4)."""

    ks_pvalue: float = 0.05
    psi_drift: float = 0.2
    psi_warning: float = 0.1


@dataclass(frozen=True)
class DriftConfig:
    """Resolved drift-detection configuration."""

    thresholds: DriftThresholds
    min_samples: int = 30
    psi_bins: int = 10
    cooldown_seconds: int = 900
    per_node: dict | None = None

    def thresholds_for(self, node: str) -> DriftThresholds:
        """Return thresholds for ``node``, applying any per-node override."""
        if not self.per_node or node not in self.per_node:
            return self.thresholds
        override = self.per_node[node] or {}
        base = self.thresholds
        return DriftThresholds(
            ks_pvalue=float(override.get("ks_pvalue", base.ks_pvalue)),
            psi_drift=float(override.get("psi_drift", base.psi_drift)),
            psi_warning=float(override.get("psi_warning", base.psi_warning)),
        )


def load_drift_config(path: Path | None = None) -> DriftConfig:
    """Load :class:`DriftConfig` from ``config/drift.yaml`` (or an explicit path)."""
    cfg_path = path or get_paths().drift_yaml
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    thr = raw.get("thresholds", {}) or {}
    det = raw.get("detection", {}) or {}
    alerting = raw.get("alerting", {}) or {}
    return DriftConfig(
        thresholds=DriftThresholds(
            ks_pvalue=float(thr.get("ks_pvalue", 0.05)),
            psi_drift=float(thr.get("psi_drift", 0.2)),
            psi_warning=float(thr.get("psi_warning", 0.1)),
        ),
        min_samples=int(det.get("min_samples", 30)),
        psi_bins=int(det.get("psi_bins", 10)),
        cooldown_seconds=int(alerting.get("cooldown_seconds", 900)),
        per_node=raw.get("per_node") or {},
    )
