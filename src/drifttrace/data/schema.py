"""Declared data schema model and loader (FR-2).

A dependency-light schema (per FR-2.4, decision D-3): a small typed model plus a
YAML loader. Each column declares a type, nullability, and either a numeric range
(``min`` / ``max``) or an allowed ``categories`` set.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, model_validator

from drifttrace.config import get_paths

ColumnType = Literal["float", "int", "category"]


class ColumnSchema(BaseModel):
    """Schema for a single column."""

    name: str
    type: ColumnType
    nullable: bool = False
    min: float | None = None
    max: float | None = None
    categories: list[Any] | None = None

    @model_validator(mode="after")
    def _check_constraints(self) -> ColumnSchema:
        if self.type == "category" and self.categories is None:
            raise ValueError(f"column '{self.name}': category type requires 'categories'")
        if self.min is not None and self.max is not None and self.min > self.max:
            raise ValueError(f"column '{self.name}': min {self.min} > max {self.max}")
        return self


class DataSchema(BaseModel):
    """The full declared schema: an ordered collection of column schemas."""

    columns: list[ColumnSchema] = Field(default_factory=list)

    @property
    def column_names(self) -> list[str]:
        return [c.name for c in self.columns]

    def column(self, name: str) -> ColumnSchema:
        for c in self.columns:
            if c.name == name:
                return c
        raise KeyError(name)


def load_schema(path: Path | None = None) -> DataSchema:
    """Load the declared schema from ``config/schema.yaml`` (or an explicit path)."""
    schema_path = path or get_paths().schema_yaml
    raw = yaml.safe_load(schema_path.read_text(encoding="utf-8"))
    if not raw or "columns" not in raw:
        raise ValueError(f"schema file {schema_path} has no 'columns' section")
    columns = [ColumnSchema(name=name, **spec) for name, spec in raw["columns"].items()]
    return DataSchema(columns=columns)
