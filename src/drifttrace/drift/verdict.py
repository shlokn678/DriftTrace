"""Shared drift verdict vocabulary (FR-9).

Fixed, documented verdicts used by the KS/PSI detectors, the per-node engine, RCA,
and reports. No hidden scoring: verdicts are derived only from KS p-value and PSI
against configured thresholds.
"""

from __future__ import annotations

from enum import StrEnum


class Verdict(StrEnum):
    """Per-node / per-test drift verdict."""

    STABLE = "STABLE"
    WARNING = "WARNING"
    DRIFT = "DRIFT"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"
