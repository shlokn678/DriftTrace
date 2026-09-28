"""Unit tests for the prediction event schema (FR-7.4, FR-13)."""

from __future__ import annotations

import pytest

from drifttrace.streaming.event import SCHEMA_VERSION, PredictionEvent, new_event_id


@pytest.mark.unit
def test_event_has_required_fields() -> None:
    ev = PredictionEvent(
        model_version="1",
        features={"feature_a": 1000.0, "feature_b": 600.0, "feature_c": 0.3},
        prediction=0,
        probability=0.3,
    )
    assert ev.schema_version == SCHEMA_VERSION
    assert ev.event_id
    assert ev.model_version == "1"
    assert ev.timestamp  # ISO-8601 default
    assert set(ev.features) == {"feature_a", "feature_b", "feature_c"}
    assert ev.prediction == 0
    assert ev.source == "api"


@pytest.mark.unit
def test_event_json_roundtrip() -> None:
    ev = PredictionEvent(event_id="abc", model_version="2", prediction=1, probability=0.9)
    raw = ev.to_json()
    back = PredictionEvent.from_json(raw)
    assert back == ev


@pytest.mark.unit
def test_event_bytes_roundtrip() -> None:
    ev = PredictionEvent(event_id="xyz", prediction=1, probability=0.7)
    back = PredictionEvent.from_json(ev.to_bytes())
    assert back.event_id == "xyz"


@pytest.mark.unit
def test_new_event_id_unique() -> None:
    assert new_event_id() != new_event_id()
