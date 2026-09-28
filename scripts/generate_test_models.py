"""Generate five reusable DriftTrace test-model bundles.

Produces genuinely trained scikit-learn models with real held-out metrics, plus
reference / stable / drift CSVs and (for two of them) a graph.json, packaged as
``*.drift.zip`` bundles under ``artifacts/test_models/`` (git-ignored).

This script does NOT touch the DriftTrace production implementation. It only reads the
public bundle/adapter contracts to produce artifacts that work with the current app.

Run:  python scripts/generate_test_models.py
Deterministic, offline, repository-relative paths, safe to re-run.
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import (
    load_breast_cancer,
    load_digits,
    load_wine,
    make_classification,
    make_regression,
)
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:  # cloudpickle matches the DriftTrace loader's default reader
    import cloudpickle as _pickler
except ImportError:  # pragma: no cover
    import pickle as _pickler

SEED = 42
OUT_ROOT = Path(__file__).resolve().parents[1] / "artifacts" / "test_models"


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------
def _round_df(df: pd.DataFrame, digits: int = 6) -> pd.DataFrame:
    return df.round(digits)


def _shift(df: pd.DataFrame, features: list[str], *, k: float, rng: np.random.Generator) -> pd.DataFrame:
    """Return a copy of ``df`` with a deterministic mean+scale shift on ``features``.

    The shift is proportional to each feature's own std (k * std added to the mean and a
    modest scale increase), which reliably pushes PSI above the drift threshold without
    touching DriftTrace's thresholds.
    """
    out = df.copy()
    for f in features:
        col = out[f].to_numpy(dtype=float)
        std = float(np.nanstd(col)) or 1.0
        # deterministic shift + slight spread; jitter is seeded and small.
        jitter = rng.normal(0.0, std * 0.05, size=col.shape[0])
        out[f] = col + k * std + jitter
    return out


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    _round_df(df).to_csv(path, index=False)


def _save_model(model: object, path: Path) -> None:
    with path.open("wb") as fh:
        _pickler.dump(model, fh)


def _make_bundle_zip(model_dir: Path, name: str, with_graph: bool) -> Path:
    """Zip model.pkl + reference.csv (+ graph.json) into ``<name>.drift.zip``."""
    zip_path = model_dir / f"{name}.drift.zip"
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(model_dir / "model.pkl", "model.pkl")
        zf.write(model_dir / "reference.csv", "reference.csv")
        if with_graph:
            zf.write(model_dir / "graph.json", "graph.json")
    zip_path.write_bytes(buf.getvalue())
    return zip_path


def _clf_metrics(y_true, y_pred, y_score, *, multiclass: bool) -> dict:
    m: dict = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
    if multiclass:
        m["precision_macro"] = float(precision_score(y_true, y_pred, average="macro"))
        m["recall_macro"] = float(recall_score(y_true, y_pred, average="macro"))
        m["f1_macro"] = float(f1_score(y_true, y_pred, average="macro"))
        m["f1_weighted"] = float(f1_score(y_true, y_pred, average="weighted"))
        try:
            m["roc_auc_ovr"] = float(roc_auc_score(y_true, y_score, multi_class="ovr", average="macro"))
        except Exception:  # noqa: BLE001
            m["roc_auc_ovr"] = None
    else:
        m["precision"] = float(precision_score(y_true, y_pred))
        m["recall"] = float(recall_score(y_true, y_pred))
        m["f1"] = float(f1_score(y_true, y_pred))
        try:
            m["roc_auc"] = float(roc_auc_score(y_true, y_score[:, 1]))
        except Exception:  # noqa: BLE001
            m["roc_auc"] = None
    return m


def _chain_edges(chain: list[str]) -> list[list[str]]:
    """Consecutive [parent, child] edges for an ordered dependency chain.

    ``["a", "b", "c"]`` -> ``[["a", "b"], ["b", "c"]]``. The DriftTrace graph loader
    appends the model-output node automatically (leaf features feed ``prediction``), so
    the declared chain plus the drift set below yields exactly one upstream root.
    """
    return [[chain[i], chain[i + 1]] for i in range(len(chain) - 1)]


def _emit(
    name: str,
    model: object,
    feature_frame: pd.DataFrame,
    metrics: dict,
    *,
    task: str,
    algorithm: str,
    dataset: str,
    chain: list[str] | None = None,
    drift_features: list[str] | None = None,
) -> dict:
    """Write model.pkl, reference/stable/drift CSVs, optional graph.json, metrics, zip.

    Graph-enabled models pass an ordered ``chain`` of 3-4 real feature names. The graph
    is built ONLY from that chain (a single connected dependency path), and the
    controlled drift is applied to EXACTLY those chain features so every drifted node
    lies on the chain - producing one upstream root cause + downstream symptoms under
    the unchanged RCA engine. Unrelated features stay within the normal distribution.

    No-graph models pass ``chain=None`` and an explicit ``drift_features`` list.
    """
    with_graph = chain is not None
    if with_graph:
        # Enforce graph <-> drift coupling: drift the chain features and nothing else.
        graph_edges: list[list[str]] | None = _chain_edges(chain)
        drift_features = list(chain)
    else:
        graph_edges = None
        drift_features = list(drift_features or [])

    model_dir = OUT_ROOT / name
    model_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    # Reference = a representative normal sample of FEATURE columns only (no target).
    ref = feature_frame.reset_index(drop=True)
    _write_csv(ref, model_dir / "reference.csv")

    # Stable = a fresh normal sample from the same distribution (seeded bootstrap).
    stable_idx = rng.integers(0, len(ref), size=min(300, len(ref)))
    stable = ref.iloc[stable_idx].reset_index(drop=True)
    _write_csv(stable, model_dir / "stable.csv")

    # Drift = a shifted sample on the chosen features (deterministic).
    drift_idx = rng.integers(0, len(ref), size=min(300, len(ref)))
    drift_base = ref.iloc[drift_idx].reset_index(drop=True)
    drift = _shift(drift_base, drift_features, k=3.0, rng=rng)
    _write_csv(drift, model_dir / "drift.csv")

    _save_model(model, model_dir / "model.pkl")

    with_graph = graph_edges is not None
    if with_graph:
        (model_dir / "graph.json").write_text(
            json.dumps({"edges": graph_edges}, indent=2), encoding="utf-8"
        )

    feature_names = list(ref.columns)
    metrics_out = {
        "model": name,
        "algorithm": algorithm,
        "task": task,
        "dataset": dataset,
        "n_features": len(feature_names),
        "feature_names": feature_names,
        "graph": with_graph,
        "graph_chain": list(chain) if with_graph else None,
        "graph_edges": graph_edges,
        "drift_features": drift_features,
        "metrics": metrics,
    }
    (model_dir / "metrics.json").write_text(json.dumps(metrics_out, indent=2), encoding="utf-8")

    zip_path = _make_bundle_zip(model_dir, name, with_graph)
    metrics_out["bundle"] = str(zip_path.relative_to(OUT_ROOT.parents[1]))
    print(f"  [{name}] task={task} n_features={len(feature_names)} graph={with_graph} -> {zip_path.name}")
    return metrics_out


# --------------------------------------------------------------------------------------
# model builders
# --------------------------------------------------------------------------------------
def build_breast_cancer() -> dict:
    data = load_breast_cancer(as_frame=True)
    X = data.data.copy()
    y = data.target.astype(int)
    X.columns = [str(c).replace(" ", "_") for c in X.columns]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=SEED, stratify=y)
    model = Pipeline(
        [("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=5000, random_state=SEED))]
    ).fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    y_score = model.predict_proba(X_te)
    metrics = _clf_metrics(y_te, y_pred, y_score, multiclass=False)
    # One small connected dependency chain over related size features. The generator
    # derives the edges + drifts exactly these features (one root -> symptoms).
    return _emit(
        "breast_cancer_logistic",
        model,
        X,
        metrics,
        task="classification",
        algorithm="Pipeline(StandardScaler, LogisticRegression)",
        dataset="sklearn.load_breast_cancer",
        chain=["mean_radius", "mean_perimeter", "mean_area"],
    )


def build_wine() -> dict:
    data = load_wine(as_frame=True)
    X = data.data.copy()
    y = data.target.astype(int)
    X.columns = [str(c).replace(" ", "_") for c in X.columns]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=SEED, stratify=y)
    model = RandomForestClassifier(n_estimators=300, random_state=SEED).fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    y_score = model.predict_proba(X_te)
    metrics = _clf_metrics(y_te, y_pred, y_score, multiclass=True)
    return _emit(
        "wine_random_forest",
        model,
        X,
        metrics,
        task="classification",
        algorithm="RandomForestClassifier",
        dataset="sklearn.load_wine",
        drift_features=["alcohol", "color_intensity", "proline"],
    )


def build_digits() -> dict:
    data = load_digits(as_frame=True)
    X = data.data.copy()
    y = data.target.astype(int)
    X.columns = [str(c).replace(" ", "_") for c in X.columns]
    # Drop all-constant pixel columns (they carry no distribution to profile).
    non_constant = [c for c in X.columns if X[c].nunique() > 1]
    X = X[non_constant]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=SEED, stratify=y)
    model = ExtraTreesClassifier(n_estimators=300, random_state=SEED).fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    y_score = model.predict_proba(X_te)
    metrics = _clf_metrics(y_te, y_pred, y_score, multiclass=True)
    # Shift a handful of informative central pixels for the drift scenario.
    variances = X.var().sort_values(ascending=False)
    drift_feats = [str(c) for c in variances.index[:5]]
    return _emit(
        "digits_extra_trees",
        model,
        X,
        metrics,
        task="classification",
        algorithm="ExtraTreesClassifier",
        dataset="sklearn.load_digits",
        drift_features=drift_feats,
    )


def build_synthetic_classifier() -> dict:
    feature_names = [
        "customer_age",
        "monthly_usage",
        "support_calls",
        "account_duration",
        "payment_ratio",
        "service_score",
        "engagement_score",
        "risk_indicator",
    ]
    Xarr, y = make_classification(
        n_samples=2000,
        n_features=len(feature_names),
        n_informative=6,
        n_redundant=1,
        n_classes=2,
        random_state=SEED,
    )
    X = pd.DataFrame(Xarr, columns=feature_names)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=SEED, stratify=y)
    model = GradientBoostingClassifier(random_state=SEED).fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    y_score = model.predict_proba(X_te)
    metrics = _clf_metrics(y_te, y_pred, y_score, multiclass=False)
    return _emit(
        "synthetic_gradient_boosting",
        model,
        X,
        metrics,
        task="classification",
        algorithm="GradientBoostingClassifier",
        dataset="sklearn.make_classification",
        chain=["engagement_score", "service_score", "risk_indicator"],
    )


def build_synthetic_regressor() -> dict:
    feature_names = [
        "sensor_a",
        "sensor_b",
        "sensor_c",
        "sensor_d",
        "sensor_e",
        "sensor_f",
        "sensor_g",
        "sensor_h",
    ]
    Xarr, y = make_regression(
        n_samples=2000,
        n_features=len(feature_names),
        n_informative=6,
        noise=8.0,
        random_state=SEED,
    )
    X = pd.DataFrame(Xarr, columns=feature_names)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=SEED)
    model = GradientBoostingRegressor(random_state=SEED).fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    rmse = float(np.sqrt(mean_squared_error(y_te, y_pred)))
    metrics = {
        "r2": float(r2_score(y_te, y_pred)),
        "mae": float(mean_absolute_error(y_te, y_pred)),
        "rmse": rmse,
    }
    return _emit(
        "synthetic_regression",
        model,
        X,
        metrics,
        task="regression",
        algorithm="GradientBoostingRegressor",
        dataset="sklearn.make_regression",
        drift_features=["sensor_a", "sensor_b", "sensor_c"],
    )


def _write_readme(results: list[dict]) -> None:
    lines = [
        "# DriftTrace test-model bundles",
        "",
        "Five genuinely trained scikit-learn bundles for exercising the current DriftTrace",
        "application (model-agnostic upload -> inspect -> activate -> predict -> drift/RCA).",
        "All metrics below are measured on held-out test data. These are TEST artifacts,",
        "generated by `scripts/generate_test_models.py`, and are git-ignored.",
        "",
        "| Model | Algorithm | Task | Acc | F1 / macro-F1 | ROC-AUC | R2 | Graph |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in results:
        m = r["metrics"]
        acc = f'{m.get("accuracy"):.3f}' if "accuracy" in m else "-"
        f1 = (
            f'{m.get("f1"):.3f}'
            if "f1" in m
            else (f'{m.get("f1_macro"):.3f}' if "f1_macro" in m else "-")
        )
        auc = (
            f'{m.get("roc_auc"):.3f}'
            if m.get("roc_auc") is not None
            else (f'{m.get("roc_auc_ovr"):.3f}' if m.get("roc_auc_ovr") is not None else "-")
        )
        r2 = f'{m.get("r2"):.3f}' if "r2" in m else "-"
        lines.append(
            f'| {r["model"]} | {r["algorithm"]} | {r["task"]} | {acc} | {f1} | {auc} | {r2} '
            f'| {"YES" if r["graph"] else "NO"} |'
        )
    lines += ["", "## Details", ""]
    for r in results:
        lines += [
            f'### {r["model"]}',
            f'- dataset: `{r["dataset"]}`',
            f'- features ({r["n_features"]}): {", ".join(r["feature_names"][:12])}'
            + (" ..." if r["n_features"] > 12 else ""),
            f'- graph: {"YES" if r["graph"] else "NO"}'
            + (f' (chain: {" -> ".join(r["graph_chain"])} -> prediction)' if r["graph"] else ""),
            f'- controlled drift features: {", ".join(r["drift_features"])}'
            + (" (exactly the graph chain)" if r["graph"] else ""),
            f'- bundle: `{r["bundle"]}`',
            f'- stable data: `artifacts/test_models/{r["model"]}/stable.csv`',
            f'- drift data: `artifacts/test_models/{r["model"]}/drift.csv`',
            "",
        ]
    (OUT_ROOT / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"Generating test-model bundles into {OUT_ROOT} ...")
    results = [
        build_breast_cancer(),
        build_wine(),
        build_digits(),
        build_synthetic_classifier(),
        build_synthetic_regressor(),
    ]
    (OUT_ROOT / "model_metrics.json").write_text(
        json.dumps({r["model"]: r for r in results}, indent=2), encoding="utf-8"
    )
    _write_readme(results)

    # Basic structural validation of every bundle.
    ok = True
    for r in results:
        name = r["model"]
        zip_path = OUT_ROOT / name / f"{name}.drift.zip"
        with zipfile.ZipFile(zip_path) as zf:
            names = set(zf.namelist())
        required = {"model.pkl", "reference.csv"}
        has_required = required <= names
        has_graph = "graph.json" in names
        graph_ok = has_graph == bool(r["graph"])
        status = "OK" if (has_required and graph_ok) else "BAD"
        if status != "OK":
            ok = False
        print(f"  validate {name}: {status} (contents={sorted(names)})")
    print("\nDone." if ok else "\nDone WITH STRUCTURAL ISSUES.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
