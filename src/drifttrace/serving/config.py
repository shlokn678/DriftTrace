"""Environment-driven configuration for the serving layer (NFR-4, NFR-7).

All settings come from environment variables with safe local defaults; no secrets and
no machine-specific absolute paths are hard-coded. The defaults target direct local
execution (localhost); an optional, separately-run Redpanda broker can be pointed at via
``REDPANDA_BROKER`` but is not required for the normal workflow.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from drifttrace.config import get_paths


def _env_bool(name: str, default: bool = False) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    val = os.environ.get(name)
    if val is None:
        return default
    try:
        return int(val)
    except ValueError:
        return default


@dataclass
class ServingSettings:
    """Resolved serving configuration."""

    use_redpanda: bool
    redpanda_brokers: str
    topic: str
    event_log_path: Path
    reports_dir_path: Path
    cors_allow_origins: list[str]
    # Live-monitoring settings (additive). Defaults keep the existing paths unchanged;
    # live monitoring is opt-in via POST /live/start.
    live_source: str = "file"
    live_window_size: int = 100
    live_min_window_samples: int = 30
    live_incident_open_after: int = 2
    live_incident_close_after: int = 3


def get_serving_settings() -> ServingSettings:
    """Build :class:`ServingSettings` from the environment."""
    paths = get_paths()
    return ServingSettings(
        use_redpanda=_env_bool("DRIFTTRACE_USE_REDPANDA", default=False),
        redpanda_brokers=os.environ.get("REDPANDA_BROKER", "localhost:9092"),
        topic=os.environ.get("DRIFTTRACE_TOPIC", "drifttrace.predictions"),
        event_log_path=Path(
            os.environ.get("DRIFTTRACE_EVENT_LOG", str(paths.reports / "events.jsonl"))
        ),
        reports_dir_path=Path(os.environ.get("DRIFTTRACE_REPORTS_DIR", str(paths.reports))),
        cors_allow_origins=[
            o.strip()
            for o in os.environ.get(
                "DRIFTTRACE_CORS_ORIGINS",
                "http://localhost:5173,http://localhost:4173,http://127.0.0.1:5173",
            ).split(",")
            if o.strip()
        ],
        # Live monitoring: default to the broker-free file source so the demo runs with
        # no Kafka install; set DRIFTTRACE_LIVE_SOURCE=redpanda to use a running broker.
        live_source=os.environ.get(
            "DRIFTTRACE_LIVE_SOURCE", "redpanda" if _env_bool("DRIFTTRACE_USE_REDPANDA") else "file"
        ),
        live_window_size=_env_int("DRIFTTRACE_LIVE_WINDOW_SIZE", 100),
        live_min_window_samples=_env_int("DRIFTTRACE_LIVE_MIN_WINDOW_SAMPLES", 30),
        live_incident_open_after=_env_int("DRIFTTRACE_LIVE_INCIDENT_OPEN_AFTER", 2),
        live_incident_close_after=_env_int("DRIFTTRACE_LIVE_INCIDENT_CLOSE_AFTER", 3),
    )
