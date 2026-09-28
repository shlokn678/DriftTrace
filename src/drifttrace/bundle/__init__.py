"""Model bundle domain (model-agnostic).

A DriftTrace *bundle* is what a user uploads to monitor any supported model:

    my_model.drift.zip
    ├── model.pkl        (required)  the trained model
    ├── reference.csv    (required)  the baseline / reference data for that model
    └── graph.json       (optional)  feature dependency edges for stronger RCA

There is no built-in model and no domain-specific assumption. The reference data
defines the feature schema and the drift baseline for its own model; the optional
graph enables dependency-based root-cause tracing.
"""

from __future__ import annotations

from drifttrace.bundle.graph_json import GraphParseError, graph_from_edges, load_graph_json
from drifttrace.bundle.loader import (
    BundleError,
    LoadedBundle,
    extract_bundle_zip,
    load_bundle_dir,
)
from drifttrace.bundle.reference import ReferenceProfile, profile_reference

__all__ = [
    "BundleError",
    "LoadedBundle",
    "extract_bundle_zip",
    "load_bundle_dir",
    "ReferenceProfile",
    "profile_reference",
    "GraphParseError",
    "graph_from_edges",
    "load_graph_json",
]
