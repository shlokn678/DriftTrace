"""Phase 0 smoke test: the package is importable and core deps are present."""

import importlib

import pytest


@pytest.mark.unit
def test_package_imports_and_has_version() -> None:
    mod = importlib.import_module("drifttrace")
    assert hasattr(mod, "__version__")
    assert mod.__version__ == "0.1.0"


@pytest.mark.unit
@pytest.mark.parametrize(
    "name",
    ["numpy", "pandas", "scipy", "sklearn", "networkx", "pydantic", "yaml", "structlog"],
)
def test_core_dependencies_importable(name: str) -> None:
    importlib.import_module(name)
