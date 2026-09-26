"""Monitoring report generation (FR-17).

Produces a machine-readable JSON report and a human-readable Markdown summary for one
monitoring window, tying the drift results, RCA verdict, and alert to the five
lifecycle-evidence anchors (dataset version, code commit, pipeline execution, model
artifact/version, monitoring report id).

Reports clearly distinguish ROOT_CAUSE / SYMPTOM / STABLE / WARNING / INSUFFICIENT_DATA.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from drifttrace.drift.engine import DriftReport
from drifttrace.graph.dag import DependencyGraph
from drifttrace.rca.engine import RCAResult, classify_nodes


def new_report_id() -> str:
    return "rpt-" + uuid.uuid4().hex[:12]


@dataclass
class MonitoringReport:
    """A complete monitoring report for one window."""

    report_id: str
    window_id: str
    incident_id: str | None
    model_version: str
    baseline_version: str
    dataset_version: str | None
    code_commit: str | None
    timestamp: str
    classifications: dict[str, str]
    drift: dict
    rca: dict
    alerts: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    def to_markdown(self) -> str:
        lines: list[str] = []
        lines.append(f"# DriftTrace Monitoring Report {self.report_id}")
        lines.append("")
        lines.append(f"- Window: `{self.window_id}`")
        if self.incident_id:
            lines.append(f"- Incident: `{self.incident_id}`")
        lines.append(f"- Model version: `{self.model_version}`")
        lines.append(f"- Baseline version: `{self.baseline_version}`")
        lines.append(f"- Dataset version: `{self.dataset_version or 'n/a'}`")
        lines.append(f"- Code commit: `{self.code_commit or 'n/a'}`")
        lines.append(f"- Timestamp: {self.timestamp}")
        lines.append("")
        lines.append("## Node classifications")
        lines.append("")
        lines.append("| Node | Classification | KS verdict | PSI verdict |")
        lines.append("| --- | --- | --- | --- |")
        nodes = self.drift.get("nodes", {})
        for node, cls in self.classifications.items():
            res = nodes.get(node, {})
            ks_v = (res.get("ks") or {}).get("verdict", "-")
            psi_v = (res.get("psi") or {}).get("verdict", "-")
            lines.append(f"| {node} | {cls} | {ks_v} | {psi_v} |")
        lines.append("")
        lines.append("## Root-cause analysis")
        lines.append("")
        if self.rca.get("has_root_cause"):
            for cand in self.rca.get("root_cause_candidates", []):
                path = " -> ".join([cand["node"], *cand["symptom_path"]])
                lines.append(
                    f"- **ROOT CAUSE**: `{cand['node']}` (severity {cand['severity']:.4f})"
                )
                lines.append(f"  - symptom path: `{path}`")
            if self.rca.get("symptoms"):
                lines.append(f"- Symptoms (evidence only): {self.rca['symptoms']}")
        else:
            lines.append("- No root cause detected; no root-cause alert emitted.")
        lines.append("")
        lines.append("## Alerts")
        lines.append("")
        if self.alerts:
            for a in self.alerts:
                lines.append(
                    f"- `{a['incident_id']}` root_cause=`{a['root_cause']}` "
                    f"delivery=`{a['delivery_status']}`"
                )
        else:
            lines.append("- None.")
        lines.append("")
        return "\n".join(lines)

    def save(self, reports_dir: Path) -> dict[str, Path]:
        reports_dir.mkdir(parents=True, exist_ok=True)
        json_path = reports_dir / f"{self.report_id}.json"
        md_path = reports_dir / f"{self.report_id}.md"
        latest_json = reports_dir / "latest_rca.json"
        json_path.write_text(self.to_json(), encoding="utf-8")
        md_path.write_text(self.to_markdown(), encoding="utf-8")
        latest_json.write_text(self.to_json(), encoding="utf-8")
        return {"json": json_path, "markdown": md_path, "latest": latest_json}


def build_report(
    drift: DriftReport,
    rca: RCAResult,
    graph: DependencyGraph,
    *,
    model_version: str,
    alerts: list[dict] | None = None,
    dataset_version: str | None = None,
    code_commit: str | None = None,
    incident_id: str | None = None,
) -> MonitoringReport:
    """Assemble a :class:`MonitoringReport` from drift + RCA + alert results."""
    return MonitoringReport(
        report_id=new_report_id(),
        window_id=drift.window_id,
        incident_id=incident_id,
        model_version=model_version,
        baseline_version=drift.baseline_version,
        dataset_version=dataset_version,
        code_commit=code_commit,
        timestamp=datetime.now(UTC).isoformat(),
        classifications=classify_nodes(drift, graph),
        drift=drift.to_dict(),
        rca=rca.to_dict(),
        alerts=alerts or [],
    )
