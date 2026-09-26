"""Environment-driven configuration for the serving layer (NFR-4, NFR-7).

All settings come from environment variables with safe local defaults; no secrets and
no machine-specific absolute paths are hard-coded. Container-to-container communication
uses Docker service names (e.g. ``redpanda:9092``, ``http://mlflow:5000``) supplied via
the environment in Compose.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from drifttrace.config import get_paths
from drifttrace.features.transform import TransformParams


def _env_bool(name: str, default: bool = False) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class ServingSettings:
    """Resolved serving configuration."""

    mlflow_tracking_uri: str | None
    model_version: str | None
    use_redpanda: bool
    redpanda_brokers: str
    topic: str
    event_log_path: Path
    transform_params_path: Path
    reports_dir_path: Path

    def load_transform_params(self) -> TransformParams:
        """Load transform params persisted by the training pipeline (FR-3.3).

        Falls back to identity-ish defaults only if the file is missing, so the service
        can still start (and be exercised) in environments where training has not been
        run; in the normal flow the file is present next to the model artifacts.
        """
        if self.transform_params_path.exists():
            data = json.loads(self.transform_params_path.read_text(encoding="utf-8"))
            return TransformParams(
                income_ref_log_mean=float(data["income_ref_log_mean"]),
                income_ref_log_std=float(data["income_ref_log_std"]),
                credit_coef=float(data.get("credit_coef", 1.0)),
                risk_credit_coef=float(data.get("risk_credit_coef", -1.5)),
            )
        # Safe fallback (log-space around a typical income); real runs use the file.
        return TransformParams(income_ref_log_mean=8.5, income_ref_log_std=0.5)


def get_serving_settings() -> ServingSettings:
    """Build :class:`ServingSettings` from the environment."""
    paths = get_paths()
    return ServingSettings(
        mlflow_tracking_uri=os.environ.get("MLFLOW_TRACKING_URI"),
        model_version=os.environ.get("DRIFTTRACE_MODEL_VERSION"),
        use_redpanda=_env_bool("DRIFTTRACE_USE_REDPANDA", default=False),
        redpanda_brokers=os.environ.get("REDPANDA_BROKER", "localhost:9092"),
        topic=os.environ.get("DRIFTTRACE_TOPIC", "drifttrace.predictions"),
        event_log_path=Path(
            os.environ.get("DRIFTTRACE_EVENT_LOG", str(paths.reports / "events.jsonl"))
        ),
        transform_params_path=Path(
            os.environ.get(
                "DRIFTTRACE_TRANSFORM_PARAMS",
                str(paths.artifacts / "transform_params.json"),
            )
        ),
        reports_dir_path=Path(os.environ.get("DRIFTTRACE_REPORTS_DIR", str(paths.reports))),
    )
