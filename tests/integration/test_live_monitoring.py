"""Integration tests for the live-monitoring extension.

Drives the REAL live pipeline (existing TumblingWindower + Phase4Processor drift/RCA +
Alerter) over a broker-free file source, so CI needs no Kafka. Proves the required flow:

    stable windows -> drift windows -> RCA root cause + symptoms -> incident OPEN
    -> return to stable -> incident CLOSES

Also checks the /live/* API endpoints and that existing endpoints are unaffected.
"""

from __future__ import annotations

import io
import json
import zipfile

import cloudpickle
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression


# --------------------------------------------------------------------------------------
# Bundle: a graph-enabled classifier with a 3-node dependency chain a -> b -> c.
# --------------------------------------------------------------------------------------
def _chain_bundle() -> tuple[bytes, pd.DataFrame]:
    rng = np.random.default_rng(0)
    n = 500
    df = pd.DataFrame(
        {
            "feat_a": rng.normal(10, 2, n),
            "feat_b": rng.normal(20, 3, n),
            "feat_c": rng.normal(30, 4, n),
            "feat_d": rng.normal(5, 1, n),
        }
    )
    y = (df["feat_a"] + df["feat_b"] > 30).astype(int)
    model = LogisticRegression(max_iter=500).fit(df, y)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        mb = io.BytesIO()
        cloudpickle.dump(model, mb)
        zf.writestr("model.pkl", mb.getvalue())
        zf.writestr("reference.csv", df.to_csv(index=False))
        zf.writestr(
            "graph.json", json.dumps({"edges": [["feat_a", "feat_b"], ["feat_b", "feat_c"]]})
        )
    return buf.getvalue(), df


@pytest.fixture
def active_ctx(tmp_path, monkeypatch):
    """Activate a graph-enabled model in a temp root; return (ctx, reference_frame)."""
    monkeypatch.setenv("DRIFTTRACE_ROOT", str(tmp_path))
    for sub in ("data", "artifacts", "reports"):
        (tmp_path / sub).mkdir(parents=True, exist_ok=True)

    from drifttrace.serving.onboarding import REGISTRY

    REGISTRY._models.clear()
    REGISTRY.deactivate()

    data, ref = _chain_bundle()
    bundle_path = tmp_path / "bundle.zip"
    bundle_path.write_bytes(data)
    onboarded = REGISTRY.onboard_bundle(bundle_path, original_name="bundle.zip")
    ctx = REGISTRY.activate(onboarded.model_id)
    return ctx, ref


def _write_events(
    path,
    frame: pd.DataFrame,
    n: int,
    *,
    shift: dict[str, float] | None = None,
    exact: bool = False,
):
    """Append n standardized prediction events built from frame (optionally shifted).

    ``exact=True`` emits reference rows in order (cycling), which reproduces the baseline
    distribution tightly - used for the recovery phase so stable windows are unambiguous.
    Otherwise rows are bootstrap-sampled (used for the shifted-drift phase).
    """
    from drifttrace.streaming.event import PredictionEvent

    rng = np.random.default_rng(123)
    path.parent.mkdir(parents=True, exist_ok=True)
    # In exact mode, draw each feature independently from its full reference column so
    # every window is a representative sample of the baseline (not a contiguous slice,
    # which can legitimately differ enough to trip PSI). This yields unambiguously
    # stable recovery windows for a deterministic incident-close assertion.
    cols = {c: frame[c].to_numpy(dtype=float) for c in frame.columns}
    with path.open("a", encoding="utf-8") as fh:
        for _ in range(n):
            if exact:
                feats = {c: float(rng.choice(cols[c])) for c in frame.columns}
            else:
                row = frame.iloc[int(rng.integers(0, len(frame)))]
                feats = {c: float(row[c]) for c in frame.columns}
            if shift:
                for f, k in shift.items():
                    feats[f] = feats[f] + k
            ev = PredictionEvent(features=feats, source="test")
            fh.write(ev.to_json() + "\n")


# --------------------------------------------------------------------------------------
# Core pipeline: stable -> drift -> RCA -> incident -> close
# --------------------------------------------------------------------------------------
@pytest.mark.integration
def test_live_pipeline_stable_then_drift_then_recover(active_ctx, tmp_path) -> None:
    from drifttrace.streaming.live import IncidentPolicy, LiveConfig, LiveMonitor, TailingFileSource
    from drifttrace.streaming.live_store import LiveStore

    ctx, ref = active_ctx
    event_log = tmp_path / "reports" / "events.jsonl"

    # Pre-write the full event stream BEFORE starting, then tail from the beginning.
    # Window size 100: 2 stable windows, 3 drifted (chain-shifted) windows, 3 stable.
    std = {f: float(ref[f].std()) for f in ("feat_a", "feat_b", "feat_c")}
    _write_events(event_log, ref, 200)  # windows 1-2 STABLE
    _write_events(
        event_log,
        ref,
        300,
        shift={
            "feat_a": 6 * std["feat_a"],
            "feat_b": 6 * std["feat_b"],
            "feat_c": 6 * std["feat_c"],
        },
    )  # windows 3-5 DRIFT (whole chain shifted, large so it is unambiguous)
    # Recovery: emit reference rows in order so the baseline is reproduced tightly and
    # these windows are unambiguously stable (>=3 clean windows to close the incident).
    _write_events(event_log, ref, 500, exact=True)  # windows 6-10 STABLE

    store = LiveStore(tmp_path / "reports" / "live")
    # Tail from the start of the file with a short idle timeout so the bounded stream ends.
    source = TailingFileSource(event_log, poll_interval=0.01, idle_timeout=0.5)
    source.read = _read_from_start(source)  # type: ignore[method-assign]

    monitor = LiveMonitor(
        ctx,
        source,
        store,
        live_config=LiveConfig(
            window_size=100,
            min_window_samples=30,
            incident=IncidentPolicy(open_after=2, close_after=3),
        ),
        reports_dir=tmp_path / "reports",
        source_label="file",
    )
    monitor.start()
    _wait_until(lambda: not monitor.is_running(), timeout=10.0)

    windows = store.read_windows()
    assert len(windows) >= 8, f"expected >=8 windows, got {len(windows)}"

    statuses = [w["status"] for w in windows]
    # Reference-consistent traffic must NOT be flagged as DRIFT (no false positives).
    # (A borderline WARNING is acceptable from resampling; DRIFT is not.)
    assert statuses[0] in ("STABLE", "WARNING")
    assert statuses[1] in ("STABLE", "WARNING")
    assert windows[0]["drifted_features"] == []
    assert windows[1]["drifted_features"] == []
    # The chain-shift windows must be flagged DRIFT by the existing engine.
    drift_windows = [w for w in windows if w["status"] == "DRIFT"]
    assert len(drift_windows) >= 2

    # RCA on a drifted window: feat_a is the upstream root; b, c are symptoms.
    rca_window = next(w for w in drift_windows if w["has_root_cause"])
    assert rca_window["primary_root_cause"] == "feat_a"
    assert set(rca_window["symptoms"]) >= {"feat_b", "feat_c"}
    assert rca_window["dependencies_available"] is True
    # Per-feature KS/PSI evidence is retained for the drifted features.
    assert rca_window["per_feature"]["feat_a"]["psi"] is not None
    assert rca_window["per_feature"]["feat_a"]["ks_p_value"] is not None

    # An incident opened during the drift stretch and closed after recovery.
    incidents = store.read_incidents()
    assert len(incidents["history"]) >= 1, "expected the incident to close after recovery"
    closed = incidents["history"][0]
    assert closed["status"] == "CLOSED"
    assert closed["root_cause"] == "feat_a"
    # Last-incident details are retained AFTER close: the advanced panels can still show
    # the most recent (closed) incident's root cause / symptom path without claiming it is
    # active. (The history entry carries the diagnosis even though the incident is closed.)
    last = incidents["history"][-1]
    assert last["root_cause"] == "feat_a"
    if last["symptom_path"]:
        assert set(last["symptom_path"]) <= {"feat_a", "feat_b", "feat_c"}


@pytest.mark.integration
def test_live_pipeline_no_graph_graceful(tmp_path, monkeypatch) -> None:
    """A no-graph model still detects drift but reports no symptom chain."""
    monkeypatch.setenv("DRIFTTRACE_ROOT", str(tmp_path))
    for sub in ("data", "artifacts", "reports"):
        (tmp_path / sub).mkdir(parents=True, exist_ok=True)

    from drifttrace.serving.onboarding import REGISTRY
    from drifttrace.streaming.live import LiveConfig, LiveMonitor, TailingFileSource
    from drifttrace.streaming.live_store import LiveStore

    REGISTRY._models.clear()
    REGISTRY.deactivate()

    rng = np.random.default_rng(1)
    n = 500
    df = pd.DataFrame(
        {"x1": rng.normal(0, 1, n), "x2": rng.normal(5, 2, n), "x3": rng.normal(9, 3, n)}
    )
    y = (df["x1"] > 0).astype(int)
    model = LogisticRegression(max_iter=300).fit(df, y)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        mb = io.BytesIO()
        cloudpickle.dump(model, mb)
        zf.writestr("model.pkl", mb.getvalue())
        zf.writestr("reference.csv", df.to_csv(index=False))  # NO graph.json
    bundle_path = tmp_path / "ng.zip"
    bundle_path.write_bytes(buf.getvalue())
    ctx = REGISTRY.activate(REGISTRY.onboard_bundle(bundle_path).model_id)
    assert ctx.graph is None

    event_log = tmp_path / "reports" / "events.jsonl"
    _write_events(event_log, df, 100, shift={"x1": 10.0, "x2": 10.0})

    store = LiveStore(tmp_path / "reports" / "live")
    source = TailingFileSource(event_log, poll_interval=0.01, idle_timeout=0.5)
    source.read = _read_from_start(source)  # type: ignore[method-assign]
    monitor = LiveMonitor(
        ctx,
        source,
        store,
        live_config=LiveConfig(window_size=100, min_window_samples=30),
        reports_dir=tmp_path / "reports",
    )
    monitor.start()
    _wait_until(lambda: not monitor.is_running(), timeout=10.0)

    windows = store.read_windows()
    assert windows, "expected at least one window"
    w = windows[0]
    assert w["dependencies_available"] is False
    assert len(w["drifted_features"]) > 0
    # No dependency graph -> no downstream symptom chain fabricated.
    assert w["symptoms"] == []


# --------------------------------------------------------------------------------------
# API surface: /live/* endpoints + existing endpoints unaffected
# --------------------------------------------------------------------------------------
@pytest.mark.integration
def test_live_api_endpoints(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("DRIFTTRACE_ROOT", str(tmp_path))
    # Pin the live-monitoring settings this test asserts on so it is hermetic and does not
    # depend on any ambient DRIFTTRACE_LIVE_* environment the shell may already carry.
    monkeypatch.setenv("DRIFTTRACE_LIVE_SOURCE", "file")
    monkeypatch.setenv("DRIFTTRACE_LIVE_WINDOW_SIZE", "100")
    monkeypatch.setenv("DRIFTTRACE_LIVE_MIN_WINDOW_SAMPLES", "30")
    monkeypatch.setenv("DRIFTTRACE_USE_REDPANDA", "false")
    for sub in ("data", "artifacts", "reports"):
        (tmp_path / sub).mkdir(parents=True, exist_ok=True)

    from fastapi.testclient import TestClient

    from drifttrace.serving.app import create_app
    from drifttrace.serving.config import get_serving_settings
    from drifttrace.serving.onboarding import REGISTRY

    REGISTRY._models.clear()
    REGISTRY.deactivate()
    client = TestClient(create_app(get_serving_settings()))

    # Status works with no model (STOPPED, not a crash).
    assert client.get("/live/status").json()["status"] in ("STOPPED", "ERROR")
    # Start with no active model -> 409.
    assert client.post("/live/start").status_code == 409

    # Activate a model, then the live endpoints respond cleanly.
    data, _ = _chain_bundle()
    mid = client.post("/models/upload", files={"file": ("b.zip", data, "application/zip")}).json()[
        "model_id"
    ]
    client.post(f"/models/{mid}/activate")

    started = client.post("/live/start").json()
    assert started["status"] in ("RUNNING", "STOPPED", "ERROR")
    assert started["model_id"] == mid
    assert started["window_size"] == 100

    # Ingest a few events through the broker-free endpoint.
    for _ in range(5):
        r = client.post(
            "/live/ingest",
            json={"features": {"feat_a": 10, "feat_b": 20, "feat_c": 30, "feat_d": 5}},
        )
        assert r.json()["accepted"] is True

    assert "windows" in client.get("/live/windows").json()
    assert "available" in client.get("/live/latest").json()
    assert "open" in client.get("/live/incidents").json()

    stopped = client.post("/live/stop").json()
    assert stopped["status"] in ("STOPPED", "ERROR")

    # Existing endpoints are unaffected by the live layer.
    assert client.get("/health").json()["status"] == "ok"
    assert client.get("/ready").json()["ready"] is True


@pytest.mark.integration
def test_batch_drift_test_still_works_with_live_layer(tmp_path, monkeypatch) -> None:
    """The existing batch/on-demand drift test must remain fully functional."""
    monkeypatch.setenv("DRIFTTRACE_ROOT", str(tmp_path))
    for sub in ("data", "artifacts", "reports"):
        (tmp_path / sub).mkdir(parents=True, exist_ok=True)

    from fastapi.testclient import TestClient

    from drifttrace.serving.app import create_app
    from drifttrace.serving.config import get_serving_settings
    from drifttrace.serving.onboarding import REGISTRY

    REGISTRY._models.clear()
    REGISTRY.deactivate()
    client = TestClient(create_app(get_serving_settings()))

    data, _ = _chain_bundle()
    mid = client.post("/models/upload", files={"file": ("b.zip", data, "application/zip")}).json()[
        "model_id"
    ]
    client.post(f"/models/{mid}/activate")
    res = client.post("/demo/run-drift-test", json={"intensity": 2.0, "n": 200, "seed": 7}).json()
    assert res["outcome"] is not None
    assert len(res["outcome"]["drifted_nodes"]) > 0


# --------------------------------------------------------------------------------------
# State isolation: no active model / model switch must not show stale current state
# --------------------------------------------------------------------------------------
def _named_bundle(cols: list[str], *, graph: bool = True) -> bytes:
    """A small graph-enabled (or no-graph) classifier bundle over the given feature names."""
    rng = np.random.default_rng(0)
    n = 400
    df = pd.DataFrame({c: rng.normal(10 * (i + 1), 3, n) for i, c in enumerate(cols)})
    y = (df[cols[0]] > float(df[cols[0]].mean())).astype(int)
    model = LogisticRegression(max_iter=300).fit(df, y)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        mb = io.BytesIO()
        cloudpickle.dump(model, mb)
        zf.writestr("model.pkl", mb.getvalue())
        zf.writestr("reference.csv", df.to_csv(index=False))
        if graph and len(cols) >= 3:
            zf.writestr(
                "graph.json",
                json.dumps({"edges": [[cols[0], cols[1]], [cols[1], cols[2]]]}),
            )
    return buf.getvalue()


@pytest.fixture
def iso_client(tmp_path, monkeypatch):
    """TestClient over a temp root with the broker-free file source and small windows."""
    monkeypatch.setenv("DRIFTTRACE_ROOT", str(tmp_path))
    monkeypatch.setenv("DRIFTTRACE_LIVE_SOURCE", "file")
    monkeypatch.setenv("DRIFTTRACE_LIVE_WINDOW_SIZE", "50")
    monkeypatch.setenv("DRIFTTRACE_LIVE_MIN_WINDOW_SAMPLES", "30")
    monkeypatch.setenv("DRIFTTRACE_USE_REDPANDA", "false")
    for sub in ("data", "artifacts", "reports"):
        (tmp_path / sub).mkdir(parents=True, exist_ok=True)

    from fastapi.testclient import TestClient

    from drifttrace.serving.app import create_app
    from drifttrace.serving.config import get_serving_settings
    from drifttrace.serving.onboarding import REGISTRY

    REGISTRY._models.clear()
    REGISTRY.deactivate()
    return TestClient(create_app(get_serving_settings()))


def _activate(client, data: bytes) -> str:
    mid = client.post("/models/upload", files={"file": ("b.zip", data, "application/zip")}).json()[
        "model_id"
    ]
    client.post(f"/models/{mid}/activate")
    return mid


def _run_live_window(client, cols: list[str], *, n: int = 120) -> None:
    """Start a live session and ingest n drifted events so a FULL window is evaluated.

    Waits for a window with at least the configured minimum samples (not merely the
    trailing partial-window flush) so callers observe a real drift evaluation with
    populated per-feature KS/PSI rather than an INSUFFICIENT_DATA partial.
    """
    import time

    client.post("/live/start")
    big = dict.fromkeys(cols, 500.0)
    for _ in range(n):
        client.post("/live/ingest", json={"features": big})
    # Give the file-tailing monitor time to form + evaluate a full (>=min-samples) window.
    for _ in range(60):
        windows = client.get("/live/windows").json()["windows"]
        if any(w["event_count"] >= 30 for w in windows):
            break
        time.sleep(0.1)
    client.post("/live/stop")


@pytest.mark.integration
def test_no_active_model_reports_no_current_drift(iso_client) -> None:
    """1 & 2: with no active model, there is no current drift and no current root cause."""
    # Run a model + live window, then deactivate -> no active model.
    cols = ["age_x", "usage_x", "calls_x"]
    _activate(iso_client, _named_bundle(cols))
    _run_live_window(iso_client, cols)
    iso_client.post("/models/deactivate")

    # /live/status: explicit no-active-model, no current result.
    status = iso_client.get("/live/status").json()
    assert status["active_model"] is False
    assert status["events_received"] == 0
    assert status["latest_window"] is None
    assert status["open_incident"] is None

    # /live/latest: nothing current.
    latest = iso_client.get("/live/latest").json()
    assert latest["available"] is False
    assert latest["window"] is None

    # /live/incidents: no current incident.
    inc = iso_client.get("/live/incidents").json()
    assert inc["open"] is None

    # /rca/latest: no current drift / root cause attributed to a (missing) active model.
    rca = iso_client.get("/rca/latest").json()
    assert rca["available"] is False
    assert rca.get("active_model") is False
    assert "report" not in rca


@pytest.mark.integration
def test_no_active_model_scopes_windows_but_preserves_history(iso_client) -> None:
    """6: historical windows remain persisted but are not shown as current state."""
    cols = ["age_x", "usage_x", "calls_x"]
    _activate(iso_client, _named_bundle(cols))
    _run_live_window(iso_client, cols)
    iso_client.post("/models/deactivate")

    # Scoped (default) windows -> none, because no active model.
    scoped = iso_client.get("/live/windows").json()
    assert scoped["windows"] == []
    assert scoped["active_model"] is False

    # Historical records are preserved on disk and retrievable with all_history.
    history = iso_client.get("/live/windows?all_history=true").json()["windows"]
    assert len(history) >= 1
    assert all("model_id" in w and "session_id" in w for w in history)


@pytest.mark.integration
def test_live_window_scoped_to_producing_model(iso_client) -> None:
    """3: a live window is tagged with the model that produced it."""
    cols = ["age_a", "usage_a", "calls_a"]
    mid = _activate(iso_client, _named_bundle(cols))
    _run_live_window(iso_client, cols)

    status = iso_client.get("/live/status").json()
    assert status["active_model"] is True
    assert status["model_id"] == mid
    latest = iso_client.get("/live/latest").json()["window"]
    assert latest is not None
    assert latest["model_id"] == mid
    assert latest["session_id"] is not None


@pytest.mark.integration
def test_switching_models_does_not_leak_previous_model_state(iso_client) -> None:
    """4: Model A's live result must not appear as Model B's current result."""
    a_cols = ["age_a", "usage_a", "calls_a"]
    mA = _activate(iso_client, _named_bundle(a_cols))
    _run_live_window(iso_client, a_cols)
    # Confirm A has a current window.
    assert iso_client.get("/live/latest").json()["window"]["model_id"] == mA

    # Activate a different Model B (different feature names), no live session yet.
    b_cols = ["temp_b", "humid_b", "press_b"]
    mB = _activate(iso_client, _named_bundle(b_cols))

    status = iso_client.get("/live/status").json()
    assert status["model_id"] == mB
    assert status["active_model"] is True
    assert status["events_received"] == 0
    assert status["latest_window"] is None  # A's window must not leak into B

    assert iso_client.get("/live/latest").json()["available"] is False
    assert iso_client.get("/live/windows").json()["windows"] == []  # scoped to B -> none yet
    assert iso_client.get("/live/incidents").json()["open"] is None
    # /rca/latest must not attribute A's drift report to B.
    assert iso_client.get("/rca/latest").json()["available"] is False

    # A's windows remain in the full history, tagged with A's id.
    history = iso_client.get("/live/windows?all_history=true").json()["windows"]
    assert any(w["model_id"] == mA for w in history)


@pytest.mark.integration
def test_rca_latest_not_shown_without_active_model_after_run(iso_client) -> None:
    """5: stale batch RCA result is not presented as current when no model is active."""
    cols = ["age_x", "usage_x", "calls_x"]
    _activate(iso_client, _named_bundle(cols))
    # Produce a real batch RCA report (writes latest_rca.json).
    res = iso_client.post("/demo/run-drift-test", json={"intensity": 2.0, "n": 150, "seed": 7})
    assert res.json()["outcome"] is not None
    # With the model active, the report is current.
    assert iso_client.get("/rca/latest").json()["available"] is True

    # Deactivate -> the previously-produced report must NOT be shown as current.
    iso_client.post("/models/deactivate")
    rca = iso_client.get("/rca/latest").json()
    assert rca["available"] is False
    assert rca.get("active_model") is False


# --------------------------------------------------------------------------------------
# Advanced-panel wiring: the live endpoints expose everything the RCA panels need
# --------------------------------------------------------------------------------------
@pytest.mark.integration
def test_live_latest_exposes_per_feature_ks_psi_for_panels(iso_client) -> None:
    """The adapted RCA panels read window.per_feature KS/PSI + RCA fields from /live/latest.

    After an evaluated drifted window on a 3-column graph bundle, every drifted feature
    must carry non-null ks_statistic, ks_p_value AND psi, and the RCA fields (primary root
    cause + symptoms) must be populated so the live panels can render real diagnostics.
    """
    cols = ["age_x", "usage_x", "calls_x"]
    _activate(iso_client, _named_bundle(cols))
    _run_live_window(iso_client, cols)

    # /live/latest (what the live panels consume) is available and carries a window.
    latest = iso_client.get("/live/latest").json()
    assert latest["available"] is True
    assert latest["window"] is not None

    # The evaluated full (>=min-samples) drifted window exposes real per-feature KS/PSI and
    # dependency-based RCA, which is exactly what the adapter feeds into the panels.
    windows = iso_client.get("/live/windows").json()["windows"]
    drifted = [w for w in windows if w["event_count"] >= 30 and w["drifted_features"]]
    assert drifted, "expected at least one full drifted window"
    window = drifted[-1]

    for feat in window["drifted_features"]:
        pf = window["per_feature"][feat]
        assert pf["ks_statistic"] is not None
        assert pf["ks_p_value"] is not None
        assert pf["psi"] is not None

    # Dependency-based RCA populated (3-column chain graph): a primary root + symptoms.
    assert window["dependencies_available"] is True
    assert window["primary_root_cause"] is not None
    assert window["symptoms"]


@pytest.mark.integration
def test_live_status_active_model_matches_while_running(iso_client) -> None:
    """/live/status reports active_model true with a matching model_id + a latest window."""
    cols = ["age_x", "usage_x", "calls_x"]
    mid = _activate(iso_client, _named_bundle(cols))
    _run_live_window(iso_client, cols)

    status = iso_client.get("/live/status").json()
    assert status["active_model"] is True
    assert status["model_id"] == mid
    assert status["latest_window"] is not None
    assert (status["windows_evaluated"] or 0) >= 1


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------
def _read_from_start(source):
    """A read() that tails the file from the beginning (so pre-written events are seen)."""
    import time

    from drifttrace.streaming.event import PredictionEvent

    def read(limit=None):
        source.path.parent.mkdir(parents=True, exist_ok=True)
        source.path.touch(exist_ok=True)
        idle = 0.0
        with source.path.open("r", encoding="utf-8") as fh:
            while not source._stop.is_set():
                line = fh.readline()
                if not line:
                    time.sleep(source.poll_interval)
                    idle += source.poll_interval
                    if source.idle_timeout is not None and idle >= source.idle_timeout:
                        return
                    continue
                idle = 0.0
                line = line.strip()
                if not line:
                    continue
                try:
                    yield PredictionEvent.from_json(line)
                except Exception:  # noqa: BLE001
                    continue

    return read


def _wait_until(predicate, timeout=10.0, interval=0.05):
    import time

    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()
