"""Governance configuration loader (FR-15)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from drifttrace.config import get_paths


@dataclass(frozen=True)
class GovernanceConfig:
    sensitive_attribute: str | None
    positive_label: int
    pii_deny_list: list[str] = field(default_factory=list)
    retention_note: str = ""


def load_governance_config(path: Path | None = None) -> GovernanceConfig:
    cfg_path = path or get_paths().governance_yaml
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    fairness = raw.get("fairness", {}) or {}
    privacy = raw.get("privacy", {}) or {}
    return GovernanceConfig(
        sensitive_attribute=fairness.get("sensitive_attribute"),
        positive_label=int(fairness.get("positive_label", 1)),
        pii_deny_list=list(privacy.get("pii_deny_list", []) or []),
        retention_note=str(privacy.get("prediction_log_retention_note", "")),
    )
