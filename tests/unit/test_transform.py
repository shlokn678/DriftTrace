"""Unit tests for the shared feature transform (FR-3)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from drifttrace.data.generator import GeneratorParams, generate
from drifttrace.features.transform import (
    CREDIT_MAX,
    CREDIT_MIN,
    MODEL_FEATURES,
    fit_params,
    transform,
)


@pytest.fixture
def fitted_params():
    df = generate(GeneratorParams(n_rows=2000, seed=11))
    return fit_params(df["income"].to_numpy(dtype=float)), df


@pytest.mark.unit
def test_transform_produces_derived_features(fitted_params) -> None:
    params, df = fitted_params
    out = transform(df[["income"]], params)
    assert set(MODEL_FEATURES).issubset(out.columns)
    assert out["credit_score"].between(CREDIT_MIN, CREDIT_MAX).all()
    assert out["risk_score"].between(0.0, 1.0).all()


@pytest.mark.unit
def test_train_serve_equivalence_batch_vs_single(fitted_params) -> None:
    """FR-3 AC-3: single-record serving equals its value inside a batch."""
    params, df = fitted_params
    batch = transform(df[["income"]], params)
    # Transform each record individually and compare.
    for idx in [0, 1, 5, 100, 999]:
        single = transform(df[["income"]].iloc[[idx]], params)
        np.testing.assert_allclose(
            single["credit_score"].to_numpy(),
            batch["credit_score"].to_numpy()[[idx]],
            rtol=1e-12,
        )
        np.testing.assert_allclose(
            single["risk_score"].to_numpy(),
            batch["risk_score"].to_numpy()[[idx]],
            rtol=1e-12,
        )


@pytest.mark.unit
def test_transform_is_deterministic(fitted_params) -> None:
    params, df = fitted_params
    a = transform(df[["income"]], params)
    b = transform(df[["income"]], params)
    pd.testing.assert_frame_equal(a, b)


@pytest.mark.unit
def test_transform_overwrites_smuggled_derived_values(fitted_params) -> None:
    """Serving cannot inject inconsistent derived features."""
    params, df = fitted_params
    tampered = df[["income"]].copy()
    tampered["credit_score"] = -1.0  # nonsense value
    out = transform(tampered, params)
    assert out["credit_score"].between(CREDIT_MIN, CREDIT_MAX).all()


@pytest.mark.unit
def test_transform_requires_income() -> None:
    params = fit_params(np.array([1000.0, 2000.0, 3000.0]))
    with pytest.raises(KeyError):
        transform(pd.DataFrame({"x": [1, 2, 3]}), params)


@pytest.mark.unit
def test_higher_income_gives_higher_credit(fitted_params) -> None:
    params, _ = fitted_params
    low = transform(pd.DataFrame({"income": [1000.0]}), params)["credit_score"].iloc[0]
    high = transform(pd.DataFrame({"income": [50000.0]}), params)["credit_score"].iloc[0]
    assert high > low
