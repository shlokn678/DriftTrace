"""Continuous live-monitoring orchestration (additive extension).

This layer ORCHESTRATES the existing DriftTrace engines for a continuously-arriving
stream of prediction events. It does NOT contain any drift or RCA logic of its own:

- Windowing          -> ``drifttrace.streaming.window.TumblingWindower`` (existing)
- Drift (KS + PSI)   -> ``drifttrace.streaming.processing.Phase4Processor`` (existing)
- RCA                -> called by Phase4Processor via ``drifttrace.rca.engine`` (existing)
- Root-cause alerts  -> ``drifttrace.alerting.alerter.Alerter`` (existing)
- Persistence        -> ``drifttrace.streaming.live_store.LiveStore`` (JSON files)

The ONLY new behaviour here is the window-to-window incident LIFECYCLE (open after N
consecutive drifted windows, close after M consecutive stable windows). This does not
redefine drift: a window's drift status comes verbatim from the existing engine's
``drifted_nodes``.

A :class:`LiveMonitor` runs the existing :class:`Monitor` loop on a background thread so
it never blocks the serving/prediction path. The broker (Redpanda) is optional: a
broker-free in-memory/file source is supported for local demos and tests, and a broker
outage surfaces as an ERROR status instead of crashing the app.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from drifttrace.alerting.alerter import Alerter
from drifttrace.drift.config import DriftConfig, load_drift_config
from drifttrace.serving.onboarding import ActiveContext
from drifttrace.streaming.event import PredictionEvent
from drifttrace.streaming.live_store import LiveStore
from drifttrace.streaming.processing import Phase4Processor, ProcessingOutcome
from drifttrace.streaming.source import EventSource
from drifttrace.streaming.window import TumblingWindower, Window, WindowPolicy

if TYPE_CHECKING:
    from drifttrace.graph.dag import DependencyGraph

logger = logging.getLogger("drifttrace.streaming.live")


# =====================================================================================
# Configuration
# =====================================================================================
@dataclass(frozen=True)
class IncidentPolicy:
    """Consecutive-window policy for the incident lifecycle.

    These control ONLY when a sequence of per-window drift verdicts is treated as one
    continuing incident. They do not change the drift thresholds or the statistical
    drift definition, which remain owned by the existing drift engine.
    """

    open_after: int = 2  # consecutive drifted windows to confirm/open an incident
    close_after: int = 3  # consecutive stable (non-drift) windows to close an incident


@dataclass(frozen=True)
class LiveConfig:
    """Resolved live-monitoring configuration."""

    window_size: int = 100
    min_window_samples: int = 30
    incident: IncidentPolicy = field(default_factory=IncidentPolicy)


# =====================================================================================
# Incident lifecycle (the only new "logic"; drift verdicts come from the engine)
# =====================================================================================
@dataclass
class Incident:
    """A continuing drift incident spanning one or more consecutive drifted windows."""

    incident_id: str
    status: str  # "OPEN" | "CLOSED"
    opened_window: str
    opened_at: str
    root_cause: str | None
    symptom_path: list[str] = field(default_factory=list)
    drifted_windows: int = 0
    last_window: str | None = None
    closed_window: str | None = None
    closed_at: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


class IncidentTracker:
    """Translate a stream of per-window drift verdicts into an incident lifecycle.

    Rule (configurable, not a new drift definition):
    - A window is "drifted" iff the existing engine reported >=1 drifted node.
    - After ``open_after`` consecutive drifted windows, an incident is OPEN.
    - While OPEN, each further drifted window updates the same incident.
    - After ``close_after`` consecutive stable windows, the OPEN incident CLOSES.
    """

    def __init__(self, policy: IncidentPolicy) -> None:
        self.policy = policy
        self.open_incident: Incident | None = None
        self.history: list[Incident] = []
        self._consecutive_drift = 0
        self._consecutive_stable = 0
        self._seq = 0

    def _new_incident_id(self) -> str:
        self._seq += 1
        return f"live-inc-{self._seq:04d}"

    def observe(
        self,
        window_id: str,
        drifted: bool,
        root_cause: str | None,
        symptom_path: list[str],
    ) -> Incident | None:
        """Record one window's drift verdict; return the open incident if any."""
        now = datetime.now(UTC).isoformat()
        if drifted:
            self._consecutive_drift += 1
            self._consecutive_stable = 0
            if self.open_incident is None:
                if self._consecutive_drift >= self.policy.open_after:
                    self.open_incident = Incident(
                        incident_id=self._new_incident_id(),
                        status="OPEN",
                        opened_window=window_id,
                        opened_at=now,
                        root_cause=root_cause,
                        symptom_path=list(symptom_path),
                        drifted_windows=self._consecutive_drift,
                        last_window=window_id,
                    )
            else:
                # Update the same continuing incident.
                self.open_incident.drifted_windows += 1
                self.open_incident.last_window = window_id
                if root_cause and not self.open_incident.root_cause:
                    self.open_incident.root_cause = root_cause
                if symptom_path and not self.open_incident.symptom_path:
                    self.open_incident.symptom_path = list(symptom_path)
        else:
            self._consecutive_stable += 1
            self._consecutive_drift = 0
            if (
                self.open_incident is not None
                and self._consecutive_stable >= self.policy.close_after
            ):
                self.open_incident.status = "CLOSED"
                self.open_incident.closed_window = window_id
                self.open_incident.closed_at = now
                self.history.append(self.open_incident)
                self.open_incident = None

        return self.open_incident

    def snapshot(self) -> dict:
        """Serializable incident state: the open incident + closed history."""
        return {
            "open": self.open_incident.to_dict() if self.open_incident else None,
            "history": [inc.to_dict() for inc in self.history],
        }


# =====================================================================================
# Window result shaping (serialize existing engine output for history / UI)
# =====================================================================================
def _window_timestamps(window: Window) -> tuple[str | None, str | None]:
    stamps = [e.timestamp for e in window.events if e.timestamp]
    if not stamps:
        return None, None
    return min(stamps), max(stamps)


def build_window_result(
    window: Window,
    outcome: ProcessingOutcome,
    *,
    sequence: int,
    dependencies_available: bool,
    incident: Incident | None,
    model_id: str | None = None,
    model_version: str | None = None,
    session_id: str | None = None,
    graph_nodes: list[str] | None = None,
    graph_edges: list[list[str]] | None = None,
) -> dict:
    """Shape one window's existing-engine output into a persisted history record.

    Every statistical value here is copied from the existing DriftReport / RCAResult -
    nothing is recomputed. Retains per-feature KS/PSI, drift status, RCA, and alert info
    so the history answers "what happened to this model over time?".

    ``model_id`` / ``model_version`` / ``session_id`` scope the record to the model and
    monitoring session that produced it, so a later process (or a different active model)
    never mistakes this window for the current active model's state.
    """
    drift = outcome.drift
    rca = outcome.rca
    start_ts, end_ts = _window_timestamps(window)

    per_feature: dict[str, dict] = {}
    for node, res in drift.nodes.items():
        ks = res.ks or {}
        psi = res.psi or {}
        per_feature[node] = {
            "verdict": res.verdict,
            "ks_statistic": ks.get("statistic"),
            "ks_p_value": ks.get("p_value"),
            "psi": psi.get("psi"),
        }

    drifted = list(drift.drifted_nodes)
    status = "DRIFT" if drifted else "STABLE"
    # Surface a WARNING summary when nothing drifted but at least one node warns.
    if not drifted and any(r.verdict == "WARNING" for r in drift.nodes.values()):
        status = "WARNING"

    roots = [c.node for c in rca.root_cause_candidates]
    primary_root = roots[0] if roots else None
    primary_symptoms = (
        rca.root_cause_candidates[0].symptom_path if rca.root_cause_candidates else []
    )

    return {
        "sequence": sequence,
        "model_id": model_id,
        "model_version": model_version,
        "session_id": session_id,
        "graph_nodes": list(graph_nodes) if graph_nodes else [],
        "graph_edges": [list(e) for e in graph_edges] if graph_edges else [],
        "window_id": outcome.window_id,
        "start_timestamp": start_ts,
        "end_timestamp": end_ts,
        "event_count": window.count,
        "status": status,
        "drifted_features": drifted,
        "drifted_count": len(drifted),
        "per_feature": per_feature,
        "dependencies_available": dependencies_available,
        "has_root_cause": rca.has_root_cause,
        "root_cause_candidates": roots,
        "primary_root_cause": primary_root,
        "symptoms": list(rca.symptoms),
        "primary_symptom_path": list(primary_symptoms),
        "alerts": outcome.alerts,
        "incident_id": incident.incident_id if incident else None,
        "incident_status": incident.status if incident else None,
        "evaluated_at": datetime.now(UTC).isoformat(),
    }


# =====================================================================================
# Live monitor (background thread orchestrating the existing Monitor loop)
# =====================================================================================
class LiveMonitor:
    """Run continuous monitoring of the active model on a background thread.

    The monitor captures the active model's context (baseline + optional graph) at
    START and keeps using it for every window - a new model is never created per window,
    and the reference baseline is never auto-updated. Stop replacing the model requires
    an explicit activate/stop by the operator.
    """

    def __init__(
        self,
        ctx: ActiveContext,
        source: EventSource,
        store: LiveStore,
        *,
        live_config: LiveConfig,
        drift_config: DriftConfig | None = None,
        reports_dir: Path | None = None,
        source_label: str = "file",
    ) -> None:
        self.ctx = ctx
        self.source = source
        self.store = store
        self.live_config = live_config
        self.drift_config = drift_config or load_drift_config()
        self.reports_dir = reports_dir
        self.source_label = source_label

        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()

        # Model + session scoping so persisted state is never mistaken for a different
        # (or absent) active model's current state.
        self.model_id: str = ctx.model_id
        meta = ctx.adapter.metadata()
        self.model_version: str | None = meta.model_version
        self.session_id: str = "live-sess-" + uuid.uuid4().hex[:12]

        # Live counters.
        self.status: str = "STOPPED"
        self.error: str | None = None
        self.events_received = 0
        self.windows_evaluated = 0
        self.current_window_count = 0
        self.started_at: str | None = None

        self._windower = TumblingWindower(
            WindowPolicy(
                size=live_config.window_size,
                min_samples=live_config.min_window_samples,
            )
        )
        self._incidents = IncidentTracker(live_config.incident)
        # Reuse the existing root-cause-only alerter; cooldown 0 so each confirmed window
        # can alert, while the incident lifecycle prevents spam at the incident level.
        self._alerter = Alerter(
            webhook=None,
            cooldown_seconds=0,
            alert_log_path=(reports_dir / "live" / "alerts.jsonl") if reports_dir else None,
        )
        self._processor = Phase4Processor(
            baseline=ctx.baseline,
            graph=ctx.graph if ctx.graph is not None else _trivial_graph(ctx.feature_names),
            config=self.drift_config,
            alerter=self._alerter if ctx.graph is not None else None,
            reports_dir=None,  # live history is persisted via LiveStore, not per-report files
            model_version=ctx.model_id,
        )

    # ---- lifecycle -------------------------------------------------------------------
    def start(self) -> None:
        """Start the background monitoring thread. Idempotent while running."""
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop.clear()
            self.status = "RUNNING"
            self.error = None
            self.events_received = 0
            self.windows_evaluated = 0
            self.current_window_count = 0
            self.started_at = datetime.now(UTC).isoformat()
            # A fresh session id per start, so a restarted session is distinct and old
            # persisted windows are never mistaken for this session's current state.
            self.session_id = "live-sess-" + uuid.uuid4().hex[:12]
            self.store.clear()
            self._persist_status()
            self._thread = threading.Thread(target=self._run, name="drifttrace-live", daemon=True)
            self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        """Signal the thread to stop and wait briefly for it to finish."""
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout)
        with self._lock:
            if self.status != "ERROR":
                self.status = "STOPPED"
            self._persist_status()

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ---- the loop --------------------------------------------------------------------
    def _run(self) -> None:
        try:
            for event in self._iter_events():
                if self._stop.is_set():
                    break
                self.events_received += 1
                self.current_window_count = (self.events_received) % self.live_config.window_size
                closed = self._windower.add(event)
                if closed is not None:
                    self._evaluate_window(closed)
                    self.current_window_count = 0
                # Persist lightweight progress periodically (every 10 events) so the UI
                # shows the current window filling without excessive disk writes.
                if self.events_received % 10 == 0:
                    self._persist_status()
            # Flush a trailing partial window when the source is exhausted/stopped so a
            # small demo still produces a final evaluation.
            tail = self._windower.flush()
            if tail is not None and tail.count > 0:
                self._evaluate_window(tail)
        except Exception as exc:  # noqa: BLE001 - a monitor failure must not crash the app
            logger.exception("live monitor failed")
            with self._lock:
                self.status = "ERROR"
                self.error = f"{type(exc).__name__}: {exc}"
        finally:
            with self._lock:
                if self.status == "RUNNING" and not self._stop.is_set():
                    # Source ended on its own (e.g. bounded file replay): mark stopped.
                    self.status = "STOPPED"
                self.current_window_count = 0
                self._persist_status()

    def _iter_events(self) -> Iterator[PredictionEvent]:
        """Yield events from the source, tolerating a broker that is unavailable."""
        try:
            yield from self.source.read()
        except Exception as exc:  # noqa: BLE001 - broker outage -> ERROR status, no crash
            logger.warning("live source read failed: %s", exc)
            with self._lock:
                self.status = "ERROR"
                self.error = f"event source unavailable: {type(exc).__name__}: {exc}"
            return

    def _evaluate_window(self, window: Window) -> None:
        # Drift + RCA + alert via the EXISTING engine (no duplication).
        outcome = self._processor.process(window)
        self.windows_evaluated += 1

        drifted = bool(outcome.drift.drifted_nodes)
        roots = [c.node for c in outcome.rca.root_cause_candidates]
        primary_root = roots[0] if roots else None
        primary_symptoms = (
            outcome.rca.root_cause_candidates[0].symptom_path
            if outcome.rca.root_cause_candidates
            else []
        )
        incident = self._incidents.observe(
            outcome.window_id, drifted, primary_root, list(primary_symptoms)
        )

        # Declared dependency order (parents before children) only when a real graph was
        # provided. For no-graph models we use the trivial graph (features independent),
        # so there is no meaningful ordering and we leave it empty.
        graph_nodes = list(self.ctx.graph.feature_nodes()) if self.ctx.graph is not None else None
        graph_edges = self.ctx.graph.edge_list() if self.ctx.graph is not None else None
        result = build_window_result(
            window,
            outcome,
            sequence=self.windows_evaluated,
            dependencies_available=self.ctx.graph is not None,
            incident=incident,
            model_id=self.model_id,
            model_version=self.model_version,
            session_id=self.session_id,
            graph_nodes=graph_nodes,
            graph_edges=graph_edges,
        )
        self.store.append_window(result)
        incidents_snapshot = self._incidents.snapshot()
        # Scope the incident record to this model + session so a later/absent active
        # model never shows this as its current incident.
        incidents_snapshot["model_id"] = self.model_id
        incidents_snapshot["model_version"] = self.model_version
        incidents_snapshot["session_id"] = self.session_id
        self.store.write_incidents(incidents_snapshot)
        self._persist_status(latest=result)

    # ---- status ----------------------------------------------------------------------
    def status_dict(self, latest: dict | None = None) -> dict:
        meta = self.ctx.adapter.metadata()
        return {
            "status": self.status,
            "error": self.error,
            "model_id": self.ctx.model_id,
            "model_version": meta.model_version,
            "model_name": meta.name,
            "session_id": self.session_id,
            "dependencies_available": self.ctx.graph is not None,
            "source": self.source_label,
            "window_size": self.live_config.window_size,
            "events_received": self.events_received,
            "current_window_count": self.current_window_count,
            "windows_evaluated": self.windows_evaluated,
            "started_at": self.started_at,
            "updated_at": datetime.now(UTC).isoformat(),
            "latest_window": latest if latest is not None else self.store.latest_window(),
            "open_incident": self._incidents.snapshot()["open"],
        }

    def _persist_status(self, latest: dict | None = None) -> None:
        try:
            self.store.write_status(self.status_dict(latest=latest))
        except OSError as exc:  # persistence failure must not crash monitoring
            logger.warning("live status persistence failed: %s", exc)


def _trivial_graph(feature_names: list[str]) -> DependencyGraph:
    """Edge-free graph (features independent) for no-graph models.

    Mirrors the serving app's no-graph handling: drift is still detected per feature, but
    no upstream root-cause tracing is possible. Kept here so the live monitor behaves
    identically to the batch path for models without a dependency graph.
    """
    from drifttrace.bundle.reference import OUTPUT_NODE
    from drifttrace.graph.dag import DependencyGraph, NodeSpec

    specs = [NodeSpec(name=f, kind="raw_input") for f in feature_names]
    specs.append(NodeSpec(name=OUTPUT_NODE, kind="model_output", parents=tuple(feature_names)))
    return DependencyGraph(specs)


# =====================================================================================
# Source construction (broker-free file source, or optional Redpanda)
# =====================================================================================
def make_live_source(
    *,
    source: str,
    event_log_path: Path,
    brokers: str,
    topic: str,
) -> tuple[EventSource, str]:
    """Build a live event source. Returns (source, label).

    - ``file``     -> a tailing JSON-lines source (broker-free; default, demo-friendly).
    - ``redpanda`` -> a continuous Kafka-API consumer (requires confluent_kafka + broker).

    A broker that is unavailable is NOT contacted here; the failure surfaces when the
    monitor thread first tries to read, and is reported as an ERROR status.
    """
    if source == "redpanda":
        return ContinuousRedpandaSource(brokers=brokers, topic=topic), "redpanda"
    return TailingFileSource(event_log_path), "file"


class ContinuousRedpandaSource:
    """A continuous Kafka-API consumer for the live monitor (additive).

    This does NOT replace or modify ``drifttrace.streaming.source.RedpandaSource`` (which
    is intentionally bounded for batch/replay use and remains unchanged). The live monitor
    needs a consumer that:
      - keeps polling until the operator stops it (no idle timeout exit), matching a
        truly continuous stream, and
      - uses a UNIQUE consumer group per session so a fresh ``/live/start`` reads new
        events from the configured offset instead of resuming a prior session's committed
        offset (which would make a repeated demo skip events).

    ``confluent_kafka`` is imported lazily so importing this module never requires the
    broker client; a broker outage raises on first read and the monitor reports ERROR.
    """

    def __init__(
        self,
        brokers: str,
        topic: str,
        *,
        group_id: str | None = None,
        poll_timeout: float = 1.0,
    ) -> None:
        import uuid

        self.brokers = brokers
        self.topic = topic
        # Unique group per session unless an explicit one is given.
        self.group_id = group_id or f"drifttrace-live-{uuid.uuid4().hex[:8]}"
        self.poll_timeout = poll_timeout
        self._consumer: Any = None
        self._stop = threading.Event()

    def stop(self) -> None:
        self._stop.set()

    def _ensure_consumer(self) -> None:
        if self._consumer is not None:
            return
        from confluent_kafka import Consumer  # lazy import

        self._consumer = Consumer(
            {
                "bootstrap.servers": self.brokers,
                "group.id": self.group_id,
                "auto.offset.reset": "latest",  # a fresh session reads events after it starts
                "enable.auto.commit": True,
            }
        )
        self._consumer.subscribe([self.topic])

    def read(self, limit: int | None = None) -> Iterator[PredictionEvent]:
        """Yield events continuously until ``stop()`` is called (or ``limit`` reached)."""
        import contextlib

        self._ensure_consumer()
        assert self._consumer is not None
        count = 0
        try:
            while not self._stop.is_set():
                msg = self._consumer.poll(self.poll_timeout)
                if msg is None or msg.error():
                    continue
                try:
                    yield PredictionEvent.from_json(msg.value())
                except Exception:  # noqa: BLE001 - skip malformed messages, keep streaming
                    continue
                count += 1
                if limit is not None and count >= limit:
                    return
        finally:
            with contextlib.suppress(Exception):
                self._consumer.close()
            self._consumer = None


class TailingFileSource:
    """A broker-free event source that follows a JSON-lines event log as it grows.

    This lets the live pipeline run end to end with NO Kafka install: the serving app's
    FileSink appends prediction events to the log, and this source tails new lines. It
    blocks (polling) until the monitor is stopped, matching a continuous stream.
    """

    def __init__(self, path: Path, poll_interval: float = 0.2, idle_timeout: float | None = None):
        self.path = Path(path)
        self.poll_interval = poll_interval
        # When set, stop after this many seconds with no new line (used by tests/demos).
        self.idle_timeout = idle_timeout
        self._stop = threading.Event()

    def stop(self) -> None:
        self._stop.set()

    def read(self, limit: int | None = None) -> Iterator[PredictionEvent]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)
        count = 0
        idle = 0.0
        with self.path.open("r", encoding="utf-8") as fh:
            # Start at the end so we only consume events produced after monitoring starts.
            fh.seek(0, 2)
            while not self._stop.is_set():
                line = fh.readline()
                if not line:
                    time.sleep(self.poll_interval)
                    idle += self.poll_interval
                    if self.idle_timeout is not None and idle >= self.idle_timeout:
                        return
                    continue
                idle = 0.0
                line = line.strip()
                if not line:
                    continue
                try:
                    yield PredictionEvent.from_json(line)
                except Exception:  # noqa: BLE001 - skip malformed lines, keep streaming
                    continue
                count += 1
                if limit is not None and count >= limit:
                    return
