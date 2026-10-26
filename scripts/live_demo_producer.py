"""Live demo producer for DriftTrace continuous monitoring.

Sends prediction events continuously so the live-monitoring pipeline fills windows and
evaluates drift over time. Two modes:

  normal : stable traffic drawn from the active model's reference distribution.
  drift  : traffic with a deliberate mean+scale shift on chosen features, so the
           EXISTING KS/PSI detector flags drift. For a graph-enabled model it shifts the
           declared dependency chain, so the EXISTING RCA engine traces one root cause.

The producer does NOT modify reference.csv and does NOT create a new model. It only
produces standardized PredictionEvents for the currently active model.

Transport (default: broker-free HTTP ingest into the running API):
  - http      : POST each event to the API's /live/ingest (no Kafka needed).
  - redpanda  : publish to the Redpanda topic via the existing RedpandaSink.

The producer is model-agnostic: it derives the feature schema, numeric/categorical
types, and (when present) the declared dependency graph from the ACTIVE model's
reference data and graph.json - never from a hardcoded feature list. Pass ``--bundle``
to derive the schema from a specific bundle (it must match the active model), and
``--drift-features`` to choose exactly which features to shift (validated against the
real schema). When neither is given, DRIFT defaults to the declared graph's upstream
chain, else the first numeric reference features.

Usage examples:
  python scripts/live_demo_producer.py --mode normal --events 300 --interval 0.02
  python scripts/live_demo_producer.py --mode drift  --events 300 --interval 0.02
  python scripts/live_demo_producer.py --mode normal --events 200 --then-drift 300
  # a different, previously-unseen bundle (schema derived from the bundle):
  python scripts/live_demo_producer.py --bundle artifacts/test_models/wine_random_forest/wine_random_forest.drift.zip --mode normal --then-drift 300
  # choose drift features explicitly (validated against the active model):
  python scripts/live_demo_producer.py --mode drift --drift-features alcohol,proline

Deterministic: a fixed seed makes a demo run repeatable.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
DEFAULT_API = "http://localhost:8000"


def _get(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=10) as resp:  # noqa: S310 - localhost demo
        return json.loads(resp.read().decode("utf-8"))


def _post(url: str, payload: dict) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(  # noqa: S310 - localhost demo
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=10) as resp:  # noqa: S310
        return json.loads(resp.read().decode("utf-8"))


def _active_model(api: str) -> dict:
    info = _get(f"{api}/models/active")
    if not info.get("active"):
        print(
            "[producer] ERROR: no active model. Upload + activate a bundle first.", file=sys.stderr
        )
        raise SystemExit(2)
    return info


def _model_store_dir(api: str) -> Path | None:
    """The local store directory for the active model's onboarded bundle, if present."""
    model_id = _active_model(api).get("model_id")
    if not model_id:
        return None
    d = REPO / "artifacts" / "uploaded_models" / str(model_id)
    return d if d.exists() else None


def _reference_frame(api: str, features: list[str]) -> pd.DataFrame | None:
    """Best-effort: locate the active model's reference.csv in the local model store.

    The producer runs locally alongside the server, so it can read the same reference
    data the server activated. If it cannot be found, synthetic normal values are used.
    """
    store = _model_store_dir(api)
    if store is None:
        return None
    path = store / "reference.csv"
    if path.exists():
        try:
            return pd.read_csv(path)
        except Exception:  # noqa: BLE001
            return None
    return None


def _graph_edges_from_store(api: str) -> list[list[str]] | None:
    """Read the active model's declared graph edges from its onboarded graph.json."""
    store = _model_store_dir(api)
    if store is None:
        return None
    return _read_graph_edges(store / "graph.json")


def _read_graph_edges(graph_path: Path) -> list[list[str]] | None:
    """Parse ``{"edges": [[parent, child], ...]}`` from a graph.json, if present."""
    if not graph_path.exists():
        return None
    try:
        data = json.loads(graph_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    edges = data.get("edges") if isinstance(data, dict) else None
    if not isinstance(edges, list):
        return None
    return [list(e) for e in edges if isinstance(e, (list, tuple)) and len(e) == 2]


def _read_bundle(bundle: Path) -> tuple[pd.DataFrame | None, list[list[str]] | None]:
    """Read (reference frame, graph edges) directly from a bundle dir or .zip.

    Used by ``--bundle`` so the producer can derive the real schema + declared graph for
    a specific bundle without depending on the model store layout. The bundle must match
    the active model (same feature schema) for the produced events to be valid.
    """
    import tempfile
    import zipfile

    if bundle.is_dir():
        ref_path = bundle / "reference.csv"
        ref = pd.read_csv(ref_path) if ref_path.exists() else None
        return ref, _read_graph_edges(bundle / "graph.json")
    if bundle.is_file() and bundle.suffix.lower() == ".zip":
        with tempfile.TemporaryDirectory(prefix="dt_bundle_") as tmp:
            tmpd = Path(tmp)
            try:
                with zipfile.ZipFile(bundle) as zf:
                    zf.extractall(tmpd)
            except zipfile.BadZipFile as exc:
                raise SystemExit(f"[producer] ERROR: not a valid .zip bundle: {exc}") from exc
            # Descend into a single subdir if the archive wrapped its files.
            entries = list(tmpd.iterdir())
            subdirs = [p for p in entries if p.is_dir()]
            root = (
                subdirs[0]
                if (len([p for p in entries if p.is_file()]) == 0 and len(subdirs) == 1)
                else tmpd
            )
            ref_path = root / "reference.csv"
            ref = pd.read_csv(ref_path) if ref_path.exists() else None
            return ref, _read_graph_edges(root / "graph.json")
    raise SystemExit(f"[producer] ERROR: --bundle path is not a directory or .zip: {bundle}")


def _feature_columns(
    ref: pd.DataFrame | None, features: list[str]
) -> tuple[np.ndarray | None, dict[str, np.ndarray], dict[str, float]]:
    """Return (reference row matrix, per-feature value arrays, per-feature std).

    NORMAL traffic is generated by bootstrapping WHOLE reference rows (the row matrix),
    which reproduces the baseline joint distribution faithfully and avoids the stochastic
    per-window false drift that independent per-feature sampling (and, worse, Gaussians
    from mean/std) can cause - especially for correlated, many-feature models at a
    100-event window. The per-feature arrays are a fallback when a full reference frame
    is unavailable; ``stds`` only sizes the DRIFT shift.
    """
    cols: dict[str, np.ndarray] = {}
    stds: dict[str, float] = {}
    for f in features:
        if ref is not None and f in ref.columns and pd.api.types.is_numeric_dtype(ref[f]):
            arr = ref[f].to_numpy(dtype=float)
            arr = arr[~np.isnan(arr)]
            cols[f] = arr if arr.size else np.zeros(1)
            stds[f] = float(np.nanstd(ref[f].to_numpy(dtype=float))) or 1.0
        else:
            cols[f] = np.zeros(1)
            stds[f] = 1.0

    rows: np.ndarray | None = None
    if ref is not None and all(
        f in ref.columns and pd.api.types.is_numeric_dtype(ref[f]) for f in features
    ):
        frame = ref[features].dropna()
        if len(frame) > 0:
            rows = frame.to_numpy(dtype=float)  # shape (n_rows, n_features), feature order
    return rows, cols, stds


def _graph_chain_from_edges(edges: list[list[str]], features: list[str]) -> list[str]:
    """Derive an ordered upstream chain of real features from declared graph edges.

    Picks the longest root-to-leaf path over the feature nodes (excluding the synthetic
    model-output node ``prediction``), so drift injected along it yields one upstream
    root cause + downstream symptoms under the EXISTING RCA engine. Nothing here is
    model-specific: the chain is read from the model's own declared graph.
    """
    feat_set = set(features)
    children: dict[str, list[str]] = {}
    parents: set[str] = set()
    all_nodes: set[str] = set()
    for pair in edges:
        if len(pair) != 2:
            continue
        p, c = str(pair[0]), str(pair[1])
        if p == "prediction" or c == "prediction":
            continue
        if p not in feat_set or c not in feat_set:
            continue
        children.setdefault(p, []).append(c)
        parents.add(p)
        all_nodes.add(p)
        all_nodes.add(c)
    if not all_nodes:
        return []
    has_parent = {c for cs in children.values() for c in cs}
    roots = [n for n in all_nodes if n not in has_parent]

    best: list[str] = []

    def walk(node: str, acc: list[str]) -> None:
        nonlocal best
        acc = [*acc, node]
        kids = children.get(node, [])
        if not kids:
            if len(acc) > len(best):
                best = acc
            return
        for k in kids:
            if k not in acc:  # guard against cycles (graph is a DAG, but be safe)
                walk(k, acc)

    for r in roots:
        walk(r, [])
    return best


def _numeric_features(ref: pd.DataFrame | None, features: list[str]) -> list[str]:
    if ref is None:
        return list(features)
    return [f for f in features if f in ref.columns and pd.api.types.is_numeric_dtype(ref[f])]


def resolve_drift_features(
    features: list[str],
    ref: pd.DataFrame | None,
    edges: list[list[str]] | None,
    explicit: list[str] | None,
) -> tuple[list[str], str]:
    """Decide which features DRIFT mode shifts, fully from the model's own schema/graph.

    Precedence (no hardcoded feature names):
      1. ``explicit`` (``--drift-features``), validated against the real schema.
      2. the declared graph's longest upstream chain (graph-enabled models).
      3. the first few numeric features from the reference schema.

    Returns (drift_features, how) where ``how`` explains the choice for the log.
    """
    numeric = _numeric_features(ref, features)
    if explicit:
        unknown = [f for f in explicit if f not in features]
        if unknown:
            raise SystemExit(
                f"[producer] ERROR: --drift-features not in the model schema: {unknown}. "
                f"available: {features}"
            )
        non_numeric = [f for f in explicit if numeric and f not in numeric]
        if non_numeric:
            print(
                f"[producer] WARNING: these --drift-features are non-numeric and the "
                f"mean+std shift may not apply cleanly: {non_numeric}",
                file=sys.stderr,
            )
        return list(explicit), "explicit --drift-features"
    if edges:
        chain = _graph_chain_from_edges(edges, features)
        if len(chain) >= 2:
            return chain, "declared graph dependency chain"
    if numeric:
        return numeric[: min(3, len(numeric))], "first numeric reference features"
    return features[: min(3, len(features))], "first reference features (no numeric info)"


def _event(
    features: list[str],
    rows: np.ndarray | None,
    cols: dict[str, np.ndarray],
    stds: dict[str, float],
    rng: np.random.Generator,
    *,
    drift: bool,
    drift_feats: list[str],
    k: float,
) -> dict:
    """Build one prediction-event payload by resampling the reference distribution.

    NORMAL: a WHOLE reference row is bootstrapped (preserving feature correlations), so a
    window of these events reproduces the baseline and does not produce stochastic false
    drift. If no full reference frame is available, fall back to per-feature resampling.
    DRIFT: the chosen drift features get a deterministic mean shift of ``k`` standard
    deviations added on top.
    """
    if rows is not None:
        row = rows[int(rng.integers(0, rows.shape[0]))]
        values = {features[i]: float(row[i]) for i in range(len(features))}
    else:
        values = {f: float(rng.choice(cols[f])) for f in features}
    if drift:
        for f in drift_feats:
            if f in values:
                values[f] = values[f] + k * stds[f]
    return {
        "features": {k2: round(v, 6) for k2, v in values.items()},
        "source": "live-demo",
        "model_version": "active",
    }


def _send(api: str, transport: str, sink, event: dict) -> None:
    if transport == "redpanda":
        from drifttrace.streaming.event import PredictionEvent

        sink.emit(PredictionEvent.model_validate(event))
    else:
        _post(f"{api}/live/ingest", event)


def _run_phase(
    api: str,
    transport: str,
    sink,
    features: list[str],
    rows: np.ndarray | None,
    cols: dict[str, np.ndarray],
    stds: dict[str, float],
    rng: np.random.Generator,
    *,
    mode: str,
    n: int,
    interval: float,
    drift_feats: list[str],
    k: float,
) -> int:
    sent = 0
    for i in range(n):
        ev = _event(
            features, rows, cols, stds, rng, drift=(mode == "drift"), drift_feats=drift_feats, k=k
        )
        try:
            _send(api, transport, sink, ev)
            sent += 1
        except urllib.error.URLError as exc:
            print(f"[producer] send failed ({exc}); is the API running?", file=sys.stderr)
            return sent
        if (i + 1) % 25 == 0:
            print(f"[producer] {mode}: sent {i + 1}/{n}")
        if interval > 0:
            time.sleep(interval)
    return sent


def main() -> int:
    ap = argparse.ArgumentParser(description="DriftTrace live demo producer")
    ap.add_argument("--api", default=DEFAULT_API, help="serving API base URL")
    ap.add_argument("--mode", choices=["normal", "drift"], default="normal")
    ap.add_argument("--events", type=int, default=300, help="events to send in the primary mode")
    ap.add_argument("--interval", type=float, default=0.02, help="seconds between events")
    ap.add_argument(
        "--then-drift",
        type=int,
        default=0,
        help="after the primary (normal) phase, send this many DRIFT events",
    )
    ap.add_argument("--k", type=float, default=3.0, help="drift shift in units of feature std")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--transport", choices=["http", "redpanda"], default="http")
    ap.add_argument("--brokers", default="localhost:9092")
    ap.add_argument("--topic", default="drifttrace.predictions")
    ap.add_argument(
        "--bundle",
        default=None,
        help=(
            "Optional path to a model bundle (.zip or directory) to derive the reference "
            "schema and declared graph from. Must match the active model's schema. When "
            "omitted, the active model's onboarded reference/graph are used (default)."
        ),
    )
    ap.add_argument(
        "--drift-features",
        default=None,
        help=(
            "Comma-separated feature names to shift in DRIFT mode (validated against the "
            "active model's real schema). When omitted, defaults are derived from the "
            "declared graph chain, else the first numeric reference features."
        ),
    )
    args = ap.parse_args()

    active = _active_model(args.api)
    features = list(active.get("features") or [])
    if not features:
        print("[producer] ERROR: active model has no features.", file=sys.stderr)
        return 2

    # Derive the reference schema + declared graph edges. --bundle reads them from a
    # specific bundle (must match the active model); otherwise use the active model's
    # onboarded store. Both are backward compatible.
    if args.bundle:
        ref, edges = _read_bundle(Path(args.bundle))
        if ref is not None:
            bundle_feats = list(ref.columns)
            extra = [f for f in bundle_feats if f not in features]
            missing = [f for f in features if f not in bundle_feats]
            if extra or missing:
                print(
                    f"[producer] ERROR: --bundle schema does not match the active model.\n"
                    f"  active-model features: {features}\n"
                    f"  bundle reference cols : {bundle_feats}",
                    file=sys.stderr,
                )
                return 2
    else:
        ref = _reference_frame(args.api, features)
        edges = _graph_edges_from_store(args.api)

    explicit = (
        [f.strip() for f in args.drift_features.split(",") if f.strip()]
        if args.drift_features
        else None
    )
    drift_feats, how = resolve_drift_features(features, ref, edges, explicit)
    rows, cols, stds = _feature_columns(ref, features)
    rng = np.random.default_rng(args.seed)

    sink = None
    if args.transport == "redpanda":
        from drifttrace.streaming.source import RedpandaSink

        sink = RedpandaSink(brokers=args.brokers, topic=args.topic)

    print(
        f"[producer] model={active.get('model_id')} features={len(features)} "
        f"graph={'yes' if active.get('dependencies_available') else 'no'} "
        f"drift_features={drift_feats} (via {how}) "
        f"shift={args.k}*std transport={args.transport}"
    )

    total = _run_phase(
        args.api,
        args.transport,
        sink,
        features,
        rows,
        cols,
        stds,
        rng,
        mode=args.mode,
        n=args.events,
        interval=args.interval,
        drift_feats=drift_feats,
        k=args.k,
    )
    if args.then_drift > 0:
        print(f"[producer] switching to DRIFT mode for {args.then_drift} events ...")
        total += _run_phase(
            args.api,
            args.transport,
            sink,
            features,
            rows,
            cols,
            stds,
            rng,
            mode="drift",
            n=args.then_drift,
            interval=args.interval,
            drift_feats=drift_feats,
            k=args.k,
        )

    if sink is not None:
        sink.flush()
    print(f"[producer] done. sent {total} events.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
