"""Deterministic windowing of prediction events (FR-13.2/13.7).

Broker-free: operates purely on :class:`PredictionEvent` objects, so it is unit
testable without Redpanda/Docker (NFR-8). Default policy is tumbling windows of a
fixed event count (size + min-samples configurable, FR-13.7).

This module does NOT compute drift. Phase 4 installs KS/PSI + RCA behind the window
handler; here a window is just a deterministic grouping of events plus a lightweight,
non-statistical summary used to prove the infrastructure carries events end to end.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

from drifttrace.streaming.event import PredictionEvent


@dataclass
class WindowPolicy:
    """Tumbling-window policy (FR-13.7)."""

    size: int = 500
    min_samples: int = 100


@dataclass
class Window:
    """A closed window of events plus a non-statistical summary.

    ``feature_means`` is a simple average per feature so tests/demos can observe that
    a simulated-drift window differs from a normal one WITHOUT any drift algorithm.
    """

    index: int
    events: list[PredictionEvent]
    sufficient: bool
    feature_means: dict[str, float] = field(default_factory=dict)

    @property
    def count(self) -> int:
        return len(self.events)

    def summary(self) -> dict:
        return {
            "window_index": self.index,
            "count": self.count,
            "sufficient": self.sufficient,
            "feature_means": self.feature_means,
            "sources": sorted({e.source for e in self.events}),
        }


def _feature_means(events: list[PredictionEvent]) -> dict[str, float]:
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    for ev in events:
        for name, value in ev.features.items():
            sums[name] = sums.get(name, 0.0) + float(value)
            counts[name] = counts.get(name, 0) + 1
    return {name: sums[name] / counts[name] for name in sums if counts[name] > 0}


class TumblingWindower:
    """Assemble events into fixed-size tumbling windows (deterministic)."""

    def __init__(self, policy: WindowPolicy | None = None) -> None:
        self.policy = policy or WindowPolicy()
        self._buffer: list[PredictionEvent] = []
        self._index = 0

    def add(self, event: PredictionEvent) -> Window | None:
        """Add one event; return a closed :class:`Window` when the window fills."""
        self._buffer.append(event)
        if len(self._buffer) >= self.policy.size:
            return self._close()
        return None

    def flush(self) -> Window | None:
        """Close a partial trailing window, if any events remain."""
        if self._buffer:
            return self._close()
        return None

    def _close(self) -> Window:
        events = self._buffer
        self._buffer = []
        window = Window(
            index=self._index,
            events=events,
            sufficient=len(events) >= self.policy.min_samples,
            feature_means=_feature_means(events),
        )
        self._index += 1
        return window


def windows_from_events(
    events: Iterator[PredictionEvent] | list[PredictionEvent],
    policy: WindowPolicy | None = None,
) -> list[Window]:
    """Convenience: turn a finite event sequence into a deterministic list of windows.

    Any trailing partial window is flushed so demos/tests over small samples still
    produce a window (marked ``sufficient=False`` when below ``min_samples``).
    """
    windower = TumblingWindower(policy)
    closed: list[Window] = []
    for ev in events:
        w = windower.add(ev)
        if w is not None:
            closed.append(w)
    tail = windower.flush()
    if tail is not None:
        closed.append(tail)
    return closed
