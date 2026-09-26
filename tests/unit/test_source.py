"""Unit tests for the event-source abstraction and file replay (FR-13.6)."""

from __future__ import annotations

import pytest

from drifttrace.streaming.event import PredictionEvent
from drifttrace.streaming.source import (
    EventSink,
    EventSource,
    FileReplaySource,
    FileSink,
    RedpandaSink,
    RedpandaSource,
)


def _events(n: int) -> list[PredictionEvent]:
    return [
        PredictionEvent(event_id=f"e{i}", prediction=i % 2, probability=0.5, source="test")
        for i in range(n)
    ]


@pytest.mark.unit
def test_file_sink_then_replay_roundtrip(tmp_path) -> None:
    path = tmp_path / "events.jsonl"
    sink = FileSink(path)
    events = _events(5)
    for ev in events:
        sink.emit(ev)

    replay = list(FileReplaySource(path).read())
    assert len(replay) == 5
    assert [e.event_id for e in replay] == [e.event_id for e in events]


@pytest.mark.unit
def test_file_replay_respects_limit(tmp_path) -> None:
    path = tmp_path / "events.jsonl"
    sink = FileSink(path)
    for ev in _events(10):
        sink.emit(ev)
    assert len(list(FileReplaySource(path).read(limit=3))) == 3


@pytest.mark.unit
def test_file_replay_skips_blank_lines(tmp_path) -> None:
    path = tmp_path / "events.jsonl"
    path.write_text(_events(1)[0].to_json() + "\n\n", encoding="utf-8")
    assert len(list(FileReplaySource(path).read())) == 1


@pytest.mark.unit
def test_file_source_satisfies_protocols(tmp_path) -> None:
    src = FileReplaySource(tmp_path / "x.jsonl")
    snk = FileSink(tmp_path / "y.jsonl")
    assert isinstance(src, EventSource)
    assert isinstance(snk, EventSink)


@pytest.mark.unit
def test_redpanda_adapters_import_without_broker_client(monkeypatch) -> None:
    """RedpandaSource/Sink construct without confluent_kafka being touched (lazy)."""
    src = RedpandaSource(brokers="redpanda:9092", topic="t")
    snk = RedpandaSink(brokers="redpanda:9092", topic="t")
    assert src.brokers == "redpanda:9092"
    assert snk.topic == "t"
    # No consumer/producer is created until read()/emit() is called.
    assert src._consumer is None
    assert snk._producer is None


@pytest.mark.unit
def test_redpanda_sink_emit_uses_lazy_producer(monkeypatch) -> None:
    """emit() constructs a producer lazily; we stub confluent_kafka to avoid a broker."""
    import sys
    import types

    produced: list[bytes] = []

    class _FakeProducer:
        def __init__(self, conf):  # noqa: ANN001
            self.conf = conf

        def produce(self, topic, value):  # noqa: ANN001
            produced.append(value)

        def poll(self, t):  # noqa: ANN001
            return 0

        def flush(self, t):  # noqa: ANN001
            return 0

    fake_mod = types.ModuleType("confluent_kafka")
    fake_mod.Producer = _FakeProducer  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "confluent_kafka", fake_mod)

    snk = RedpandaSink(brokers="redpanda:9092", topic="t")
    snk.emit(PredictionEvent(event_id="e1", prediction=1, probability=0.9))
    snk.flush()
    assert len(produced) == 1
    assert b"e1" in produced[0]
