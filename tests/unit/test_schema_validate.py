"""Unit tests for schema loading and validation (FR-2)."""

from __future__ import annotations

import pytest

from drifttrace.data.generator import GeneratorParams, generate
from drifttrace.data.schema import DataSchema, load_schema
from drifttrace.data.validate import validate


@pytest.mark.unit
def test_schema_loads_from_config() -> None:
    schema = load_schema()
    assert isinstance(schema, DataSchema)
    assert set(schema.column_names) == {"income", "credit_score", "risk_score", "group", "default"}


@pytest.mark.unit
def test_generated_data_passes_schema() -> None:
    schema = load_schema()
    df = generate(GeneratorParams(n_rows=1000, seed=2))
    report = validate(df, schema)
    assert report.passed, report.to_dict()
    assert report.n_rows == 1000


@pytest.mark.unit
def test_out_of_range_fails_and_names_column() -> None:
    schema = load_schema()
    df = generate(GeneratorParams(n_rows=200, seed=2))
    df.loc[0, "credit_score"] = 9999.0  # above max 900
    report = validate(df, schema)
    assert not report.passed
    cols = {v.column for v in report.violations}
    rules = {v.rule for v in report.violations}
    assert "credit_score" in cols
    assert "max" in rules


@pytest.mark.unit
def test_nulls_fail_when_not_nullable() -> None:
    schema = load_schema()
    df = generate(GeneratorParams(n_rows=200, seed=2))
    df.loc[0, "income"] = None
    report = validate(df, schema)
    assert not report.passed
    assert any(v.column == "income" and v.rule == "nullable" for v in report.violations)


@pytest.mark.unit
def test_bad_category_fails() -> None:
    schema = load_schema()
    df = generate(GeneratorParams(n_rows=200, seed=2))
    df.loc[0, "group"] = "Z"
    report = validate(df, schema)
    assert not report.passed
    assert any(v.column == "group" and v.rule == "categories" for v in report.violations)


@pytest.mark.unit
def test_missing_column_fails() -> None:
    schema = load_schema()
    df = generate(GeneratorParams(n_rows=50, seed=2)).drop(columns=["risk_score"])
    report = validate(df, schema)
    assert not report.passed
    assert any(v.column == "risk_score" and v.rule == "presence" for v in report.violations)


@pytest.mark.unit
def test_report_is_json_serializable() -> None:
    import json

    schema = load_schema()
    df = generate(GeneratorParams(n_rows=50, seed=2))
    report = validate(df, schema)
    json.dumps(report.to_dict())  # must not raise


@pytest.mark.unit
def test_monthly_to_annual_income_is_schema_valid() -> None:
    """FR-2 AC-4: the monthly->annual income change is NOT a schema violation."""
    schema = load_schema()
    df = generate(GeneratorParams(n_rows=500, seed=2))
    df["income"] = df["income"] * 12.0  # annualize
    report = validate(df, schema)
    # income has only a lower bound (min 0), so annualized values still pass.
    assert report.passed, report.to_dict()
