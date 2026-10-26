"""Live-monitoring API routes (additive).

Mounts the ``/live/*`` endpoints onto the existing FastAPI app without touching any
existing route. A single :class:`LiveMonitorManager` holds the one running monitor per
process. The monitor orchestrates the EXISTING drift/RCA/alert engines (see
``drifttrace.streaming.live``); these routes only start/stop it and read persisted state.

Design guarantees:
- Existing ``/predict`` and ``/demo/run-drift-test`` are never affected by this module.
- If the broker is unavailable, ``/live/start`` still returns 200 with an ERROR status
  rather than raising, and the batch/on-demand path keeps working.
- ``/live/ingest`` allows broker-free event submission so the full demo runs with no
  Kafka install (events are appended to the same event log the file source tails).
"""

from __future__ import annotations

import logging
import threading
from typing import Any

from drifttrace.serving.config import ServingSettings
from drifttrace.serving.onboarding import REGISTRY
from drifttrace.streaming.event import PredictionEvent
from drifttrace.streaming.live import (
    IncidentPolicy,
    LiveConfig,
    LiveMonitor,
    make_live_source,
)
from drifttrace.streaming.live_store import LiveStore
from drifttrace.streaming.source import FileSink

logger = logging.getLogger("drifttrace.serving.live")


class LiveMonitorManager:
    """Process-wide holder for the single live monitor + its store."""

    def __init__(self, settings: ServingSettings) -> None:
        self.settings = settings
        self._lock = threading.Lock()
        self._monitor: LiveMonitor | None = None
        self.store = LiveStore(settings.reports_dir_path / "live")

    # ---- lifecycle -------------------------------------------------------------------
    def start(self) -> dict:
        with self._lock:
            ctx = REGISTRY.active_context()
            if ctx is None:
                raise _NoActiveModel()

            # Stop a previous monitor cleanly before starting a new one.
            if self._monitor is not None and self._monitor.is_running():
                self._monitor.stop()

            live_config = LiveConfig(
                window_size=self.settings.live_window_size,
                min_window_samples=self.settings.live_min_window_samples,
                incident=IncidentPolicy(
                    open_after=self.settings.live_incident_open_after,
                    close_after=self.settings.live_incident_close_after,
                ),
            )
            source, label = make_live_source(
                source=self.settings.live_source,
                event_log_path=self.settings.event_log_path,
                brokers=self.settings.redpanda_brokers,
                topic=self.settings.topic,
            )
            self._monitor = LiveMonitor(
                ctx,
                source,
                self.store,
                live_config=live_config,
                reports_dir=self.settings.reports_dir_path,
                source_label=label,
            )
            self._monitor.start()
            return self._monitor.status_dict()

    def stop(self) -> dict:
        with self._lock:
            if self._monitor is None:
                return {"status": "STOPPED", "detail": "no live monitor running"}
            # Ask the source to stop reading so the monitor thread can exit promptly.
            # Both custom live sources (file tailer, continuous Redpanda) expose stop().
            src = getattr(self._monitor, "source", None)
            stop_fn = getattr(src, "stop", None)
            if callable(stop_fn):
                stop_fn()
            self._monitor.stop()
            return self._monitor.status_dict()

    # ---- model scoping ---------------------------------------------------------------
    @staticmethod
    def _active_model_id() -> str | None:
        """The id of the currently active model, or None when there is no active model."""
        ctx = REGISTRY.active_context()
        return ctx.model_id if ctx is not None else None

    @staticmethod
    def _no_model_status() -> dict:
        """Explicit, unambiguous no-active-model live status (no stale current state)."""
        return {
            "status": "STOPPED",
            "error": None,
            "model_id": None,
            "model_version": None,
            "model_name": None,
            "session_id": None,
            "active_model": False,
            "dependencies_available": False,
            "source": None,
            "window_size": None,
            "events_received": 0,
            "current_window_count": 0,
            "windows_evaluated": 0,
            "started_at": None,
            "updated_at": None,
            "latest_window": None,
            "open_incident": None,
            "detail": "no active model",
        }

    def _belongs_to_active(self, record: dict | None, active_id: str | None) -> bool:
        """True only when ``record`` is scoped to the current active model.

        Records without a ``model_id`` (e.g. written before this fix) are treated as NOT
        belonging to the active model, so legacy stale state is never shown as current.
        """
        if active_id is None or record is None:
            return False
        return record.get("model_id") == active_id

    # ---- reads (model-scoped) --------------------------------------------------------
    def status(self) -> dict:
        active_id = self._active_model_id()
        # No active model -> explicit no-model state regardless of what is on disk.
        if active_id is None:
            return self._no_model_status()
        # A monitor is running for the active model -> live, in-memory truth.
        if self._monitor is not None and self._monitor.model_id == active_id:
            d = self._monitor.status_dict()
            d["active_model"] = True
            return d
        # Active model, but its persisted status matches it -> serve it (survives refresh).
        persisted = self.store.read_status()
        if self._belongs_to_active(persisted, active_id):
            assert persisted is not None
            persisted["active_model"] = True
            return persisted
        # Active model with no (matching) live state yet -> clean no-current-result state.
        return {
            "status": "STOPPED",
            "error": None,
            "model_id": active_id,
            "active_model": True,
            "events_received": 0,
            "current_window_count": 0,
            "windows_evaluated": 0,
            "latest_window": None,
            "open_incident": None,
            "detail": "no live session for the active model yet",
        }

    def windows(self, limit: int | None, *, all_history: bool = False) -> list[dict]:
        """Window history.

        By default, only windows scoped to the ACTIVE model are returned (so the UI never
        shows another/absent model's windows as current). ``all_history=True`` returns the
        full persisted history (still carrying model_id/session_id for the caller to
        filter), satisfying "historical records may remain available".
        """
        rows = self.store.read_windows(limit=None)
        if not all_history:
            active_id = self._active_model_id()
            rows = [w for w in rows if active_id is not None and w.get("model_id") == active_id]
        if limit is not None and limit > 0:
            rows = rows[-limit:]
        return rows

    def latest(self) -> dict | None:
        active_id = self._active_model_id()
        if active_id is None:
            return None
        scoped = self.windows(limit=1)
        return scoped[-1] if scoped else None

    def incidents(self) -> dict:
        active_id = self._active_model_id()
        persisted = self.store.read_incidents()
        if active_id is None or persisted.get("model_id") != active_id:
            # No active model, or the persisted incidents belong to a different model.
            return {"open": None, "history": [], "active_model": active_id is not None}
        persisted["active_model"] = True
        return persisted


class _NoActiveModel(Exception):
    """Raised when live monitoring is requested with no active model."""


def register_live_routes(app: Any, settings: ServingSettings) -> LiveMonitorManager:
    """Attach ``/live/*`` endpoints to ``app`` and return the manager."""
    from fastapi import HTTPException

    manager = LiveMonitorManager(settings)
    app.state.live_manager = manager

    @app.post("/live/start")
    def live_start() -> dict:
        """Start continuous live monitoring of the active model.

        Returns the live status. A broker outage is reported via ``status == "ERROR"``
        rather than an HTTP error, so the rest of the app keeps working.
        """
        try:
            return manager.start()
        except _NoActiveModel as exc:
            raise HTTPException(status_code=409, detail="no active model to monitor") from exc
        except Exception as exc:  # noqa: BLE001 - starting must not 500 the whole app
            logger.warning("live start failed: %s", exc)
            return {"status": "ERROR", "error": f"could not start live monitoring: {exc}"}

    @app.post("/live/stop")
    def live_stop() -> dict:
        return manager.stop()

    @app.get("/live/status")
    def live_status() -> dict:
        return manager.status()

    @app.get("/live/windows")
    def live_windows(limit: int = 50, all_history: bool = False) -> dict:
        """Window history. By default scoped to the active model; ``all_history=true``
        returns the full persisted history (each row carries model_id/session_id)."""
        active = manager._active_model_id()
        return {
            "windows": manager.windows(limit=limit, all_history=all_history),
            "active_model": active is not None,
            "active_model_id": active,
        }

    @app.get("/live/latest")
    def live_latest() -> dict:
        latest = manager.latest()
        return {
            "available": latest is not None,
            "window": latest,
            "active_model": manager._active_model_id() is not None,
        }

    @app.get("/live/incidents")
    def live_incidents() -> dict:
        return manager.incidents()

    @app.post("/live/ingest")
    def live_ingest(event: dict) -> dict:
        """Broker-free event submission: append a prediction event to the live log.

        Lets the demo/tests drive the file-tailing source without Kafka. The event is
        validated against the standard :class:`PredictionEvent` schema.
        """
        try:
            pe = PredictionEvent.model_validate(event)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=422, detail=f"invalid prediction event: {exc}") from exc
        FileSink(settings.event_log_path).emit(pe)
        return {"accepted": True, "event_id": pe.event_id}

    return manager
