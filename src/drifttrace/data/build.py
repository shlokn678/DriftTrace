"""Materialize the dataset and validation report to disk (FR-1, FR-2.3).

This is the library function the DVC ``generate`` stage and the CLI wrap. It writes
a deterministic CSV plus a JSON validation report and a small dataset-version file.
"""

from __future__ import annotations

import json
from pathlib import Path

from drifttrace.config import get_paths
from drifttrace.data.generator import GeneratorParams, generate, to_csv_bytes
from drifttrace.data.schema import load_schema
from drifttrace.data.validate import validate
from drifttrace.training.evidence import dataset_version


def build_dataset(
    n_rows: int = 5000,
    seed: int = 42,
    out_dir: Path | None = None,
) -> dict:
    """Generate, validate, and persist the dataset. Returns a small manifest.

    Raises ValueError if the generated data fails its own declared schema (this
    should never happen for the default generator; it guards against regressions).
    """
    paths = get_paths()
    out = out_dir or paths.data
    out.mkdir(parents=True, exist_ok=True)

    params = GeneratorParams(n_rows=n_rows, seed=seed)
    frame = generate(params)
    csv_bytes = to_csv_bytes(frame)

    schema = load_schema()
    report = validate(frame, schema)
    if not report.passed:
        raise ValueError(f"generated data failed schema validation: {report.to_dict()}")

    dataset_path = out / "dataset.csv"
    dataset_path.write_bytes(csv_bytes)

    version = dataset_version(csv_bytes)
    manifest = {
        "dataset_path": str(dataset_path),
        "dataset_version": version,
        "n_rows": n_rows,
        "seed": seed,
        "validation": report.to_dict(),
    }
    (out / "validation_report.json").write_text(json.dumps(report.to_dict(), indent=2), "utf-8")
    (out / "dataset_version.txt").write_text(version + "\n", "utf-8")
    return manifest
