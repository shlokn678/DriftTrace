"""Event-source abstraction for prediction events (FR-13.6).

Defines a small interface with two implementations:

- :class:`FileReplaySource` - deterministic local replay from a JSON-lines file.
  Broker-free, so drift/monitor logic is testable in CI without Redpanda (FR-13.6).
- :class:`RedpandaSource` - Kafka-API consumer backed by Redpanda (FR-13.1/13.5).
  Imports ``confluent_kafka`` lazily so this module is importable without the broker
  client installed; the core monitor never depends on it (NFR-8).

Both yield :class:`PredictionEvent` objects, so the monitor is identical regardless of
source (FR-13.6 AC-5). The producer side (used by serving + demo) mirrors the sources.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from drifttrace.streaming.event import PredictionEvent

DEFAULT_TOPIC = "drifttrace.predictions"


@runtime_checkable
class EventSource(Protocol):
    """Anything that can yield prediction events."""

    def read(self, limit: int | None = None) -> Iterator[PredictionEvent]:
        """Yield events, up to ``limit`` if given."""
        ...


@runtime_checkable
class EventSink(Protocol):
    """Anything that can accept (emit) prediction events."""

    def emit(self, event: PredictionEvent) -> None:
        """Emit a single event."""
        ...


# --------------------------------------------------------------------------------------
# File replay (deterministic, broker-free)
# --------------------------------------------------------------------------------------
class FileReplaySource:
    """Read events deterministically from a JSON-lines file (one event per line)."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def read(self, limit: int | None = None) -> Iterator[PredictionEvent]:
        count = 0
        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                yield PredictionEvent.from_json(line)
                count += 1
                if limit is not None and count >= limit:
                    return


class FileSink:
    """Append events to a JSON-lines file (deterministic local emission)."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, event: PredictionEvent) -> None:
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(event.to_json() + "\n")


# --------------------------------------------------------------------------------------
# Redpanda (Kafka API) adapter - lazy import so the core stays broker-free
# --------------------------------------------------------------------------------------
class RedpandaSource:
    """Consume prediction events from a Redpanda/Kafka topic (FR-13.1/13.5).

    ``confluent_kafka`` is imported lazily inside methods so importing this module
    (and the monitor) never requires the broker client.
    """

    def __init__(
        self,
        brokers: str,
        topic: str = DEFAULT_TOPIC,
        group_id: str = "drifttrace-monitor",
        auto_offset_reset: str = "earliest",
        poll_timeout: float = 1.0,
    ) -> None:
        self.brokers = brokers
        self.topic = topic
        self.group_id = group_id
        self.auto_offset_reset = auto_offset_reset
        self.poll_timeout = poll_timeout
        self._consumer: Any = None

    def _ensure_consumer(self) -> None:
        if self._consumer is not None:
            return
        from confluent_kafka import Consumer  # lazy import (FR-13.6)

        self._consumer = Consumer(
            {
                "bootstrap.servers": self.brokers,
                "group.id": self.group_id,
                "auto.offset.reset": self.auto_offset_reset,
                "enable.auto.commit": True,
            }
        )
        self._consumer.subscribe([self.topic])

    def read(self, limit: int | None = None) -> Iterator[PredictionEvent]:
        """Poll the topic and yield events. Stops after ``limit`` or on idle timeout."""
        self._ensure_consumer()
        assert self._consumer is not None
        count = 0
        idle = 0
        max_idle = 5  # consecutive empty polls before returning (bounded for demos/tests)
        while True:
            msg = self._consumer.poll(self.poll_timeout)
            if msg is None:
                idle += 1
                if idle >= max_idle:
                    return
                continue
            if msg.error():
                idle += 1
                continue
            idle = 0
            yield PredictionEvent.from_json(msg.value())
            count += 1
            if limit is not None and count >= limit:
                return

    def close(self) -> None:
        if self._consumer is not None:
            self._consumer.close()
            self._consumer = None


class RedpandaSink:
    """Produce prediction events to a Redpanda/Kafka topic (FR-13.1)."""

    def __init__(self, brokers: str, topic: str = DEFAULT_TOPIC) -> None:
        self.brokers = brokers
        self.topic = topic
        self._producer: Any = None

    def _ensure_producer(self) -> None:
        if self._producer is not None:
            return
        from confluent_kafka import Producer  # lazy import (FR-13.6)

        self._producer = Producer({"bootstrap.servers": self.brokers})

    def emit(self, event: PredictionEvent) -> None:
        self._ensure_producer()
        assert self._producer is not None
        self._producer.produce(self.topic, value=event.to_bytes())
        self._producer.poll(0)

    def flush(self, timeout: float = 5.0) -> None:
        if self._producer is not None:
            self._producer.flush(timeout)
