"""Lightweight in-process metrics (FR-19.3, Prometheus-compatible text format).

No Prometheus/Grafana are installed (deferred stretch). This is a dependency-free
counter registry whose text output follows the Prometheus exposition format so a
future Prometheus scraper can consume it unchanged.
"""

from __future__ import annotations

import threading

# Documented metric names exposed by the API and monitor.
METRIC_NAMES = [
    "drifttrace_prediction_requests_total",
    "drifttrace_prediction_events_total",
    "drifttrace_windows_processed_total",
    "drifttrace_drift_checks_total",
    "drifttrace_nodes_checked_total",
    "drifttrace_drifted_nodes_total",
    "drifttrace_rca_results_total",
    "drifttrace_root_cause_alerts_total",
    "drifttrace_cooldown_suppressions_total",
    "drifttrace_webhook_failures_total",
]


class Metrics:
    """Thread-safe monotonic counter registry."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: dict[str, float] = dict.fromkeys(METRIC_NAMES, 0.0)

    def inc(self, name: str, amount: float = 1.0) -> None:
        with self._lock:
            self._counters[name] = self._counters.get(name, 0.0) + amount

    def get(self, name: str) -> float:
        with self._lock:
            return self._counters.get(name, 0.0)

    def snapshot(self) -> dict[str, float]:
        with self._lock:
            return dict(self._counters)

    def to_prometheus(self) -> str:
        """Render counters in Prometheus text exposition format."""
        lines: list[str] = []
        snap = self.snapshot()
        for name in METRIC_NAMES:
            value = snap.get(name, 0.0)
            lines.append(f"# TYPE {name} counter")
            lines.append(f"{name} {value}")
        return "\n".join(lines) + "\n"


# Process-wide metrics registry (shared by API routes and any in-process monitor).
METRICS = Metrics()
