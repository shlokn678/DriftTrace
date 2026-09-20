"""Unit tests for the deterministic data generator (FR-1.4, NFR-12)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from drifttrace.data.generator import (
    COLUMNS,
    CREDIT_MAX,
    CREDIT_MIN,
    GeneratorParams,
    generate,
    to_csv_bytes,
)


@pytest.mark.unit
def test_columns_and_shape() -> None:
    df = generate(GeneratorParams(n_rows=1000, seed=1))
    assert list(df.columns) == COLUMNS
    assert len(df) == 1000


@pytest.mark.unit
def test_determinism_same_seed_identical_bytes() -> None:
    a = to_csv_bytes(generate(GeneratorParams(n_rows=500, seed=7)))
    b = to_csv_bytes(generate(GeneratorParams(n_rows=500, seed=7)))
    assert a == b


@pytest.mark.unit
def test_different_seed_differs() -> None:
    a = to_csv_bytes(generate(GeneratorParams(n_rows=500, seed=7)))
    b = to_csv_bytes(generate(GeneratorParams(n_rows=500, seed=8)))
    assert a != b


@pytest.mark.unit
def test_ranges_respect_schema_bands() -> None:
    df = generate(GeneratorParams(n_rows=2000, seed=3))
    assert (df["income"] > 0).all()
    assert df["credit_score"].between(CREDIT_MIN, CREDIT_MAX).all()
    assert df["risk_score"].between(0.0, 1.0).all()
    assert set(df["default"].unique()).issubset({0, 1})
    assert set(df["group"].unique()).issubset({"A", "B"})


@pytest.mark.unit
def test_chained_feature_correlations() -> None:
    """The declared chain must show up as real statistical dependence."""
    df = generate(GeneratorParams(n_rows=5000, seed=5))
    # income -> credit_score : positive correlation
    assert np.corrcoef(df["income"], df["credit_score"])[0, 1] > 0.3
    # credit_score -> risk_score : negative correlation (lower credit -> higher risk)
    assert np.corrcoef(df["credit_score"], df["risk_score"])[0, 1] < -0.3


@pytest.mark.unit
def test_target_is_driven_by_risk() -> None:
    df = generate(GeneratorParams(n_rows=5000, seed=5))
    mean_risk_default = df.loc[df["default"] == 1, "risk_score"].mean()
    mean_risk_ok = df.loc[df["default"] == 0, "risk_score"].mean()
    # Defaulters should on average carry higher risk scores.
    assert mean_risk_default > mean_risk_ok


@pytest.mark.unit
def test_no_nulls() -> None:
    df = generate(GeneratorParams(n_rows=1000, seed=9))
    assert not df.isnull().to_numpy().any()


@pytest.mark.unit
def test_returns_dataframe() -> None:
    assert isinstance(generate(GeneratorParams(n_rows=10, seed=1)), pd.DataFrame)
