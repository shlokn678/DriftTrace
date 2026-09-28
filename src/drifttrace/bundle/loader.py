"""Load a DriftTrace model bundle from a directory or a .zip.

A bundle contains ``model.pkl`` (required), ``reference.csv`` (required) and an
optional ``graph.json``. The model file may also be ``model.joblib`` / ``.pickle``.
This module only locates and reads the files; adapter construction and reference
profiling happen in the onboarding layer so this stays dependency-light.
"""

from __future__ import annotations

import zipfile
from dataclasses import dataclass
from pathlib import Path

MODEL_NAMES = ("model.pkl", "model.joblib", "model.pickle")
REFERENCE_NAME = "reference.csv"
GRAPH_NAME = "graph.json"
MODEL_SUFFIXES = {".pkl", ".joblib", ".pickle"}


class BundleError(ValueError):
    """Raised when a bundle is missing required files or is malformed.

    Carries a user-facing ``message`` and optional developer ``detail``.
    """

    def __init__(self, message: str, detail: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.detail = detail


@dataclass
class LoadedBundle:
    """Resolved paths to a bundle's files (graph_path is None when not provided)."""

    model_path: Path
    reference_path: Path
    graph_path: Path | None


def _find_one(directory: Path, names: tuple[str, ...]) -> Path | None:
    for name in names:
        candidate = directory / name
        if candidate.is_file():
            return candidate
    return None


def _find_model(directory: Path) -> Path | None:
    """Find the model file by canonical name, else any single supported file."""
    named = _find_one(directory, MODEL_NAMES)
    if named is not None:
        return named
    # Fall back to any single file with a supported model suffix at the top level.
    candidates = [
        p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in MODEL_SUFFIXES
    ]
    if len(candidates) == 1:
        return candidates[0]
    return None


def load_bundle_dir(directory: Path) -> LoadedBundle:
    """Resolve a bundle from an already-extracted directory.

    Raises :class:`BundleError` if ``model`` or ``reference.csv`` is missing.
    """
    # If the archive extracted into a single subdirectory, descend into it.
    entries = [p for p in directory.iterdir()] if directory.is_dir() else []
    subdirs = [p for p in entries if p.is_dir()]
    files = [p for p in entries if p.is_file()]
    if not files and len(subdirs) == 1:
        directory = subdirs[0]

    model_path = _find_model(directory)
    if model_path is None:
        raise BundleError(
            "The bundle is missing a model file. Include 'model.pkl' "
            "(a .joblib or .pickle model file is also accepted)."
        )

    reference_path = directory / REFERENCE_NAME
    if not reference_path.is_file():
        raise BundleError(
            "The bundle is missing 'reference.csv'. Reference data is required to "
            "establish the drift baseline."
        )

    graph_path: Path | None = directory / GRAPH_NAME
    if graph_path is not None and not graph_path.is_file():
        graph_path = None

    return LoadedBundle(
        model_path=model_path,
        reference_path=reference_path,
        graph_path=graph_path,
    )


def extract_bundle_zip(zip_path: Path, dest_dir: Path) -> LoadedBundle:
    """Extract a bundle .zip into ``dest_dir`` and resolve its files.

    Guards against path traversal (zip-slip). Raises :class:`BundleError` on a bad
    archive or missing required files.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(zip_path) as zf:
            for member in zf.namelist():
                # Reject absolute paths and traversal.
                target = (dest_dir / member).resolve()
                if not str(target).startswith(str(dest_dir.resolve())):
                    raise BundleError(
                        "The bundle archive contains an unsafe path.",
                        detail=f"member escapes destination: {member}",
                    )
            zf.extractall(dest_dir)
    except zipfile.BadZipFile as exc:
        raise BundleError(
            "The uploaded file is not a valid .zip bundle.",
            detail=str(exc),
        ) from exc
    return load_bundle_dir(dest_dir)
