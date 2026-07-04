"""
Intermediate representation (IR) for a tabular ML pipeline.

The analyzer produces this from a live sklearn object (or from AST when the
pipeline is built imperatively, e.g. manual pandas + `for` loops). The
converter only ever sees this IR plus the raw source snippet for context —
it never has to re-derive pipeline structure itself, which is what makes the
LLM step reliable enough to be useful.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class ColumnGroup:
    name: str                      # e.g. "numeric", "categorical_low_card"
    columns: list[str]
    dtype_hint: str                # "numeric" | "categorical" | "text" | "datetime" | "id"


@dataclass
class StepIR:
    """One step of a ColumnTransformer/Pipeline, or one custom function call."""
    kind: str                      # "transformer" | "estimator" | "custom_function" | "join" | "groupby_agg"
    library: str                   # "sklearn" | "xgboost" | "lightgbm" | "catboost" | "pandas" | "custom"
    class_or_func: str             # e.g. "SimpleImputer", "OneHotEncoder", "XGBClassifier"
    params: dict[str, Any] = field(default_factory=dict)
    applies_to: Optional[str] = None      # ColumnGroup.name this step acts on, or None for whole-df/estimator
    source_snippet: Optional[str] = None  # original code, for custom_function steps the LLM must re-derive
    skrub_hint: Optional[str] = None      # analyzer's best-guess mapping, e.g. "skrub.TableVectorizer"


@dataclass
class TableIR:
    var_name: str                  # variable name in source, e.g. "products_df"
    role: str                      # "primary" | "secondary" | "target_source"
    columns: list[str]
    join_key: Optional[str] = None
    join_to: Optional[str] = None  # var_name of table this joins into
    groupby_key: Optional[str] = None
    agg_ops: dict[str, list[str]] = field(default_factory=dict)  # {col: ["mean","max"]}


@dataclass
class PipelineIR:
    task: str                       # "classification" | "regression"
    tables: list[TableIR]
    column_groups: list[ColumnGroup]
    steps: list[StepIR]
    target_column: str
    target_table: str               # var_name of the table target comes from
    estimator: StepIR
    cv_strategy: Optional[str] = None      # e.g. "StratifiedKFold(5)"
    metric: Optional[str] = None           # e.g. "roc_auc"
    ensemble_of: Optional[list["PipelineIR"]] = None  # for stacking/blending solutions
    notes: list[str] = field(default_factory=list)    # analyzer warnings (custom code, leakage risk, etc.)

    def to_dict(self) -> dict:
        import dataclasses
        return dataclasses.asdict(self)
