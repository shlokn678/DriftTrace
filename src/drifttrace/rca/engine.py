"""Graph-based root-cause analysis (FR-10).

Uses the DECLARED NetworkX dependency graph (never learned). Given the per-node drift
verdicts, classifies drifted nodes as root-cause candidates or symptoms and builds the
downstream symptom paths.

Logic (FR-10.1-10.6):
- drifted = nodes whose verdict is DRIFT.
- For each drifted node: if NO drifted ancestor -> ROOT-CAUSE CANDIDATE; else SYMPTOM.
- For each candidate: downstream drifted descendants form its symptom path toward the
  output.
- Multiple independent candidates are all reported, ranked by drift severity; no single
  arbitrary winner is forced (FR-10.5).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from drifttrace.drift.engine import DriftReport
from drifttrace.drift.verdict import Verdict
from drifttrace.graph.dag import DependencyGraph


@dataclass
class RootCauseCandidate:
    """A root-cause candidate plus its downstream symptom path and evidence."""

    node: str
    severity: float
    symptom_path: list[str] = field(default_factory=list)
    evidence: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RCAResult:
    """Root-cause analysis outcome for one window."""

    window_id: str
    baseline_version: str
    has_root_cause: bool
    root_cause_candidates: list[RootCauseCandidate] = field(default_factory=list)
    symptoms: list[str] = field(default_factory=list)
    drifted_nodes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "window_id": self.window_id,
            "baseline_version": self.baseline_version,
            "has_root_cause": self.has_root_cause,
            "root_cause_candidates": [c.to_dict() for c in self.root_cause_candidates],
            "symptoms": self.symptoms,
            "drifted_nodes": self.drifted_nodes,
        }


def _severity(node: str, report: DriftReport) -> float:
    """A transparent severity for ranking co-equal roots: prefer larger PSI, then
    smaller KS p-value. This is only a RANKING aid, never a drift decision."""
    res = report.nodes.get(node)
    if res is None:
        return 0.0
    psi = (res.psi or {}).get("psi") if res.psi else None
    if psi is not None:
        return float(psi)
    p = (res.ks or {}).get("p_value") if res.ks else None
    if p is not None:
        return float(1.0 - p)
    return 0.0


def analyze(report: DriftReport, graph: DependencyGraph) -> RCAResult:
    """Run RCA over a per-node :class:`DriftReport` using the declared graph."""
    drifted = set(report.drifted_nodes)
    result = RCAResult(
        window_id=report.window_id,
        baseline_version=report.baseline_version,
        has_root_cause=False,
        drifted_nodes=sorted(drifted),
    )
    if not drifted:
        return result

    candidates: list[RootCauseCandidate] = []
    symptoms: list[str] = []

    for node in drifted:
        # Only consider ancestors that exist in the graph.
        ancestors = graph.ancestors(node) if node in graph.nodes else set()
        drifted_ancestors = ancestors & drifted
        if drifted_ancestors:
            symptoms.append(node)
        else:
            # Root-cause candidate: build its downstream drifted symptom path.
            descendants = graph.descendants(node) if node in graph.nodes else set()
            downstream_drifted = descendants & drifted
            # Order the symptom path in topological (downstream) order.
            topo = graph.topological_order()
            path = [n for n in topo if n in downstream_drifted]
            evidence = {n: report.nodes[n].to_dict() for n in [node, *path] if n in report.nodes}
            candidates.append(
                RootCauseCandidate(
                    node=node,
                    severity=_severity(node, report),
                    symptom_path=path,
                    evidence=evidence,
                )
            )

    candidates.sort(key=lambda c: c.severity, reverse=True)
    result.root_cause_candidates = candidates
    result.symptoms = sorted(symptoms)
    result.has_root_cause = bool(candidates)
    return result


def classify_nodes(report: DriftReport, graph: DependencyGraph) -> dict[str, str]:
    """Return a node -> classification map for reporting.

    Classifications: ROOT_CAUSE, SYMPTOM, WARNING, STABLE, INSUFFICIENT_DATA.
    """
    rca = analyze(report, graph)
    roots = {c.node for c in rca.root_cause_candidates}
    symptoms = set(rca.symptoms)
    out: dict[str, str] = {}
    for node, res in report.nodes.items():
        if node in roots:
            out[node] = "ROOT_CAUSE"
        elif node in symptoms:
            out[node] = "SYMPTOM"
        elif res.verdict == str(Verdict.WARNING):
            out[node] = "WARNING"
        elif res.verdict == str(Verdict.INSUFFICIENT_DATA):
            out[node] = "INSUFFICIENT_DATA"
        else:
            out[node] = "STABLE"
    return out
