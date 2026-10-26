"""Local JSON persistence for the live-monitoring layer (additive).

The simplest reliable persistence that is already compatible with the project: plain
JSON / JSON-lines files under the reports directory (see :func:`drifttrace.config.get_paths`).
No database is introduced. The data survives frontend refreshes during a running session
because it is written to disk per window, not held only in memory.

Three artifacts under ``<reports>/live/``:
- ``status.json``   - the current monitoring session + live counters (overwritten).
- ``windows.jsonl`` - one line per completed window result (append-only history).
- ``incidents.json``- the incident lifecycle state (overwritten).

This module performs NO drift / RCA logic. It only serializes results produced by the
existing engines. It is broker-free and import-light (stdlib only), so it is safe to use
from the monitor thread and unit-testable in isolation (NFR-8).
"""

from __future__ import annotations

import json
import threading
from pathlib import Path


class LiveStore:
    """Thread-safe JSON persistence for one live-monitoring session.

    A single process-wide live monitor writes here from its background thread while the
    API reads from request threads, so every file operation is guarded by a lock.
    """

    def __init__(self, live_dir: Path) -> None:
        self.live_dir = live_dir
        self.live_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    # ---- paths -----------------------------------------------------------------------
    @property
    def status_path(self) -> Path:
        return self.live_dir / "status.json"

    @property
    def windows_path(self) -> Path:
        return self.live_dir / "windows.jsonl"

    @property
    def incidents_path(self) -> Path:
        return self.live_dir / "incidents.json"

    # ---- status ----------------------------------------------------------------------
    def write_status(self, status: dict) -> None:
        with self._lock:
            self.status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")

    def read_status(self) -> dict | None:
        with self._lock:
            if not self.status_path.exists():
                return None
            try:
                return json.loads(self.status_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return None

    # ---- window history (append-only) ------------------------------------------------
    def append_window(self, window_result: dict) -> None:
        with self._lock, self.windows_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(window_result) + "\n")

    def read_windows(self, limit: int | None = None) -> list[dict]:
        with self._lock:
            if not self.windows_path.exists():
                return []
            lines = [
                line.strip()
                for line in self.windows_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        results: list[dict] = []
        for line in lines:
            try:
                results.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        if limit is not None and limit > 0:
            return results[-limit:]
        return results

    def latest_window(self) -> dict | None:
        windows = self.read_windows(limit=1)
        return windows[-1] if windows else None

    # ---- incidents -------------------------------------------------------------------
    def write_incidents(self, incidents: dict) -> None:
        with self._lock:
            self.incidents_path.write_text(json.dumps(incidents, indent=2), encoding="utf-8")

    def read_incidents(self) -> dict:
        with self._lock:
            if not self.incidents_path.exists():
                return {"open": None, "history": []}
            try:
                return json.loads(self.incidents_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return {"open": None, "history": []}

    # ---- reset -----------------------------------------------------------------------
    def clear(self) -> None:
        """Remove all persisted live artifacts (used when a new session starts)."""
        import contextlib

        with self._lock:
            for p in (self.status_path, self.windows_path, self.incidents_path):
                with contextlib.suppress(OSError):
                    p.unlink(missing_ok=True)
