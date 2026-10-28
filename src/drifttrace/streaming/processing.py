"""Phase 4 window processing pipeline (FR-9, FR-10, FR-11, FR-17).

Ties the Phase 3 monitor to the real DriftTrace intelligence:

    window -> drift detector (KS+PSI) -> RCA -> report -> root-cause alert

This is the replacement for the Phase 3 non-statistical default handler. It is
broker-free (operates on the already-assembled window), deterministic, and produces a
persisted monitoring report per window plus root-cause-only alerts.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from drifttrace.alerting.alerter import Alerter
from drifttrace.bundle.reference import OUTPUT_NODE, output_label
from drifttrace.drift.baseline import Baseline
from drifttrace.drift.config import DriftConfig
from drifttrace.drift.engine import DriftReport, detect_drift
from drifttrace.governance.report import MonitoringReport, build_report
from drifttrace.graph.dag import DependencyGraph
from drifttrace.rca.engine import RCAResult, analyze
from drifttrace.serving.metrics import METRICS
from drifttrace.streaming.event import PredictionEvent
from drifttrace.streaming.window import Window

logger = logging.getLogger("drifttrace.streaming.processing")


def _event_class_label(ev: PredictionEvent) -> str | None:
    """The discrete class label for a classifier event, as a canonical string.

    Prefers the explicit ``prediction`` class id; falls back to ``output`` (which the
    standardized event sets to ``float(prediction)`` for classifiers). Returns None when
    neither is available, so an event with no class contributes nothing (rather than a
    fabricated value).
    """
    if ev.prediction is not None:
        return output_label(ev.prediction)
    if ev.output is not None:
        return output_label(ev.output)
    return None


@dataclass
class ProcessingOutcome:
    """Per-window Phase 4 outcome."""

    window_id: str
    drift: DriftReport
    rca: RCAResult
    report: MonitoringReport
    alerts: list[dict] = field(default_factory=list)
    report_paths: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "window_id": self.window_id,
            "drifted_nodes": self.drift.drifted_nodes,
            "has_root_cause": self.rca.has_root_cause,
            "root_cause_candidates": [c.node for c in self.rca.root_cause_candidates],
            "symptoms": self.rca.symptoms,
            "report_id": self.report.report_id,
            "alerts": self.alerts,
            "report_paths": self.report_paths,
        }


class Phase4Processor:
    """Runs drift -> RCA -> report -> alert for each completed window."""

    def __init__(
        self,
        baseline: Baseline,
        graph: DependencyGraph,
        config: DriftConfig,
        alerter: Alerter | None = None,
        reports_dir: Path | None = None,
        model_version: str | None = None,
        dataset_version: str | None = None,
        code_commit: str | None = None,
    ) -> None:
        self.baseline = baseline
        self.graph = graph
        self.config = config
        self.alerter = alerter
        self.reports_dir = reports_dir
        self.model_version = model_version or baseline.model_version
        self.dataset_version = dataset_version
        self.code_commit = code_commit
        # Continuous vs categorical node lists come from the baseline.
        self._continuous = [n for n, nb in baseline.nodes.items() if nb.kind == "continuous"]
        self._categorical = [n for n, nb in baseline.nodes.items() if nb.kind == "categorical"]

    def process(self, window: Window) -> ProcessingOutcome:
        window_id = f"w{window.index}"

        # Collect per-node values from the window's events. Numeric feature values feed
        # the continuous KS/PSI path; string values feed the categorical PSI path.
        values: dict[str, list[float]] = {n: [] for n in self._continuous}
        cats: dict[str, list[str]] = {n: [] for n in self._categorical}
        # The model-output node ("prediction") is NOT in ev.features; its value lives in
        # ev.output / ev.prediction. Collect it explicitly so output drift is monitored.
        output_is_categorical = (
            OUTPUT_NODE in self.baseline.nodes
            and self.baseline.nodes[OUTPUT_NODE].kind == "categorical"
        )
        for ev in window.events:
            for node in self._continuous:
                if node == OUTPUT_NODE:
                    continue  # handled from ev.output below, never from ev.features
                v = ev.features.get(node)
                if v is not None and not isinstance(v, str):
                    values[node].append(float(v))
            for node in self._categorical:
                if node == OUTPUT_NODE:
                    continue  # handled from ev.prediction/ev.output below
                v = ev.features.get(node)
                if v is not None:
                    cats[node].append(str(v))
            # Output node: a classifier's output is a discrete class label (categorical
            # class-frequency PSI); a regressor's output is a numeric value (continuous).
            if OUTPUT_NODE in self.baseline.nodes:
                if output_is_categorical:
                    label = _event_class_label(ev)
                    if label is not None:
                        cats.setdefault(OUTPUT_NODE, []).append(label)
                elif ev.output is not None:
                    values.setdefault(OUTPUT_NODE, []).append(float(ev.output))

        drift = detect_drift(values, cats, self.baseline, self.config, window_id=window_id)
        rca = analyze(drift, self.graph)

        # Metrics.
        METRICS.inc("drifttrace_windows_processed_total")
        METRICS.inc("drifttrace_drift_checks_total")
        METRICS.inc("drifttrace_nodes_checked_total", len(drift.nodes))
        METRICS.inc("drifttrace_drifted_nodes_total", len(drift.drifted_nodes))
        METRICS.inc("drifttrace_rca_results_total")

        # Root-cause-only alerting.
        alerts: list[dict] = []
        incident_id: str | None = None
        if self.alerter is not None and rca.has_root_cause:
            emitted = self.alerter.process(rca)
            alerts = [a.to_dict() for a in emitted]
            METRICS.inc("drifttrace_root_cause_alerts_total", len(emitted))
            suppressed = len(rca.root_cause_candidates) - len(emitted)
            if suppressed > 0:
                METRICS.inc("drifttrace_cooldown_suppressions_total", suppressed)
            for a in emitted:
                if not a.delivered and a.delivery_status.startswith("failed"):
                    METRICS.inc("drifttrace_webhook_failures_total")
            if emitted:
                incident_id = emitted[0].incident_id

        report = build_report(
            drift,
            rca,
            self.graph,
            model_version=self.model_version,
            alerts=alerts,
            dataset_version=self.dataset_version,
            code_commit=self.code_commit,
            incident_id=incident_id,
        )
        report_paths: dict[str, str] = {}
        if self.reports_dir is not None:
            saved = report.save(self.reports_dir)
            report_paths = {k: str(v) for k, v in saved.items()}

        return ProcessingOutcome(
            window_id=window_id,
            drift=drift,
            rca=rca,
            report=report,
            alerts=alerts,
            report_paths=report_paths,
        )


def run_scenario_over_events(
    events: list,
    baseline: Baseline,
    graph: DependencyGraph,
    config: DriftConfig,
    *,
    alerter: Alerter | None = None,
    reports_dir: Path | None = None,
    window_size: int = 200,
    min_window_samples: int = 30,
) -> list[ProcessingOutcome]:
    """Run the Phase 4 processor over a list of events via the windower (test/demo helper).

    Broker-free: uses in-memory windowing. Returns the per-window outcomes.
    """
    from drifttrace.streaming.window import WindowPolicy, windows_from_events

    processor = Phase4Processor(
        baseline=baseline,
        graph=graph,
        config=config,
        alerter=alerter,
        reports_dir=reports_dir,
    )
    windows = windows_from_events(
        events, WindowPolicy(size=window_size, min_samples=min_window_samples)
    )
    return [processor.process(w) for w in windows]
