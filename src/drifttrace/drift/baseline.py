"""Versioned drift baseline: the training reference distribution per node (FR-9.2/9.3).

The baseline captures, for every drift-checkable node in the declared graph, the
reference sample drawn from training data. Drift detection compares live windows
against this baseline (FR-9.2). Baselines are tied to a model version so drift is
always measured against the baseline of the deployed model (FR-9.3).

Pure and broker-free: this module only depends on numpy/pandas and the graph, so it
is unit-testable without MLflow, Docker, Airflow, or a broker (NFR-8).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from drifttrace.graph.dag import DependencyGraph


@dataclass
class NodeBaseline:
    """Reference sample and summary statistics for one node."""

    node: str
    kind: str  # "continuous" | "categorical"
    values: list[float] = field(default_factory=list)  # reference sample (continuous)
    categories: dict[str, float] = field(default_factory=dict)  # category -> proportion
    count: int = 0

    def to_dict(self) -> dict:
        return {
            "node": self.node,
            "kind": self.kind,
            "values": self.values,
            "categories": self.categories,
            "count": self.count,
        }

    @staticmethod
    def from_dict(d: dict) -> NodeBaseline:
        return NodeBaseline(
            node=d["node"],
            kind=d["kind"],
            values=list(d.get("values", [])),
            categories=dict(d.get("categories", {})),
            count=int(d.get("count", 0)),
        )


@dataclass
class Baseline:
    """A full set of node baselines for a given model version."""

    model_version: str
    nodes: dict[str, NodeBaseline] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "model_version": self.model_version,
            "nodes": {k: v.to_dict() for k, v in self.nodes.items()},
        }

    @staticmethod
    def from_dict(d: dict) -> Baseline:
        return Baseline(
            model_version=d["model_version"],
            nodes={k: NodeBaseline.from_dict(v) for k, v in d.get("nodes", {}).items()},
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @staticmethod
    def load(path: Path) -> Baseline:
        return Baseline.from_dict(json.loads(path.read_text(encoding="utf-8")))


def _is_categorical(series: pd.Series) -> bool:
    return series.dtype == object or str(series.dtype).startswith("category")


def build_baseline(
    frame: pd.DataFrame,
    graph: DependencyGraph,
    model_version: str,
    max_reference_samples: int = 5000,
) -> Baseline:
    """Build a baseline from training ``frame`` for every drift-checkable node.

    Only nodes present as columns in ``frame`` are captured (the raw input and
    derived features; the model output node has no training column here).
    """
    baseline = Baseline(model_version=model_version)
    for node in graph.feature_nodes():
        if node not in frame.columns:
            continue
        series = frame[node].dropna()
        if _is_categorical(series):
            counts = series.value_counts(normalize=True)
            baseline.nodes[node] = NodeBaseline(
                node=node,
                kind="categorical",
                categories={str(k): float(v) for k, v in counts.items()},
                count=int(series.size),
            )
        else:
            values = series.to_numpy(dtype=float)
            if values.size > max_reference_samples:
                # Deterministic subsample for a bounded, reproducible reference.
                rng = np.random.default_rng(0)
                idx = np.sort(rng.choice(values.size, size=max_reference_samples, replace=False))
                values = values[idx]
            baseline.nodes[node] = NodeBaseline(
                node=node,
                kind="continuous",
                values=[float(v) for v in values],
                count=int(series.size),
            )
    return baseline
