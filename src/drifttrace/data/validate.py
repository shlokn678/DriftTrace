"""Data validation against the declared schema (FR-2).

Produces a structured :class:`ValidationReport` naming every offending column and
rule (FR-2 AC-1). The report is JSON-serializable so it can be recorded as pipeline
execution evidence (FR-2.3).

Note (FR-2 AC-4): a monthly->annual ``income`` change is schema-valid in general;
that is a drift concern (FR-9), not a schema concern. Validation is not relied on
to catch it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import pandas as pd

from drifttrace.data.schema import ColumnSchema, DataSchema


@dataclass
class Violation:
    """A single schema violation."""

    column: str
    rule: str
    detail: str
    count: int


@dataclass
class ValidationReport:
    """Result of validating a frame against the declared schema."""

    passed: bool
    n_rows: int
    checked_columns: list[str]
    violations: list[Violation] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "n_rows": self.n_rows,
            "checked_columns": self.checked_columns,
            "violations": [asdict(v) for v in self.violations],
        }


def _is_numeric(col_type: str) -> bool:
    return col_type in {"float", "int"}


def _validate_column(series: pd.Series, spec: ColumnSchema) -> list[Violation]:
    violations: list[Violation] = []

    # Nullability.
    null_count = int(series.isnull().sum())
    if null_count > 0 and not spec.nullable:
        violations.append(
            Violation(spec.name, "nullable", f"{null_count} null values not allowed", null_count)
        )

    non_null = series.dropna()

    if spec.type == "category":
        allowed = set(spec.categories or [])
        bad_mask = ~non_null.isin(allowed)
        bad = int(bad_mask.sum())
        if bad > 0:
            violations.append(
                Violation(
                    spec.name,
                    "categories",
                    f"{bad} values outside allowed set {sorted(map(str, allowed))}",
                    bad,
                )
            )
        return violations

    if _is_numeric(spec.type):
        numeric = pd.to_numeric(non_null, errors="coerce")
        coerce_failures = int(numeric.isnull().sum())
        if coerce_failures > 0:
            violations.append(
                Violation(
                    spec.name,
                    "type",
                    f"{coerce_failures} values not coercible to {spec.type}",
                    coerce_failures,
                )
            )
        if spec.type == "int":
            valid = numeric.dropna()
            non_int = int((valid != valid.round()).sum())
            if non_int > 0:
                violations.append(
                    Violation(spec.name, "type", f"{non_int} non-integer values", non_int)
                )
        valid = numeric.dropna()
        if spec.min is not None:
            below = int((valid < spec.min).sum())
            if below > 0:
                violations.append(
                    Violation(spec.name, "min", f"{below} values below min {spec.min}", below)
                )
        if spec.max is not None:
            above = int((valid > spec.max).sum())
            if above > 0:
                violations.append(
                    Violation(spec.name, "max", f"{above} values above max {spec.max}", above)
                )
    return violations


def validate(frame: pd.DataFrame, schema: DataSchema) -> ValidationReport:
    """Validate ``frame`` against ``schema`` and return a structured report."""
    violations: list[Violation] = []

    declared = schema.column_names
    present = list(frame.columns)

    missing = [c for c in declared if c not in present]
    for col in missing:
        violations.append(Violation(col, "presence", "declared column missing from data", 0))

    for spec in schema.columns:
        if spec.name in frame.columns:
            violations.extend(_validate_column(frame[spec.name], spec))

    return ValidationReport(
        passed=len(violations) == 0,
        n_rows=int(len(frame)),
        checked_columns=declared,
        violations=violations,
    )
