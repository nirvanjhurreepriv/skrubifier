"""
Stage 1: Analyzer.

Two entry points:
  - analyze_estimator(pipeline, X, y, ...)      : introspect a *live* fitted
    or unfitted sklearn Pipeline/ColumnTransformer object (works for the
    majority of Kaggle/MLE-Bench solutions that build a sklearn Pipeline).
  - analyze_source(path)                        : AST-based fallback for
    scripts that build features imperatively (manual pandas, custom loops)
    without ever instantiating a sklearn Pipeline object. Extracts
    candidate "steps" as function/method calls and flags them as
    kind="custom_function" so the converter knows it must translate logic,
    not just re-map a class name.

Design note: we deliberately keep this analyzer conservative. Anything it
isn't confident about (custom transformers, groupby/agg chains, manual
joins) is captured as a StepIR/TableIR with `source_snippet` populated, and
a note is appended to PipelineIR.notes. The LLM converter is explicitly
instructed to treat `source_snippet`-bearing steps as ground truth to
re-implement, not as something to hallucinate a skrub equivalent for.
"""
from __future__ import annotations

import ast
import inspect
from typing import Any, Optional

from .ir import ColumnGroup, PipelineIR, StepIR, TableIR

# Best-effort mapping from common sklearn/xgboost/... classes to a skrub
# building block. Used only as a *hint* embedded in the IR — the LLM makes
# the final call, since skrub's TableVectorizer often subsumes several
# sklearn steps (imputer + encoder + scaler) at once.
SKRUB_HINTS = {
    "SimpleImputer": "skrub.TableVectorizer(...) usually subsumes this",
    "OneHotEncoder": "skrub.TableVectorizer / skrub.OneHotEncoder",
    "OrdinalEncoder": "skrub.TableVectorizer(high_cardinality_transformer=...)",
    "StandardScaler": "sklearn.preprocessing.StandardScaler via .skb.apply",
    "TargetEncoder": "skrub.TableVectorizer(high_cardinality=<encoder>) — use the `high_cardinality` kwarg (skrub 0.9+)",
    "ColumnTransformer": "skrub.TableVectorizer or explicit .skb.apply per skrub selector",
    "XGBClassifier": "keep as-is, apply via .skb.apply(XGBClassifier(...), y=y)",
    "XGBRegressor": "keep as-is, apply via .skb.apply(XGBRegressor(...), y=y)",
    "LGBMClassifier": "keep as-is, apply via .skb.apply(...)",
    "LGBMRegressor": "keep as-is, apply via .skb.apply(...)",
    "CatBoostClassifier": "keep as-is, apply via .skb.apply(...)",
    "StackingClassifier": "no direct skrub primitive: compose by chaining .skb.apply "
                           "for each base learner then a meta-learner on concatenated outputs",
    "StackingRegressor": "no direct skrub primitive: compose by chaining .skb.apply "
                          "for each base learner then a meta-learner on concatenated outputs",
    "VotingClassifier": "no direct skrub primitive: replicate each base estimator via "
                         ".skb.apply, average/vote probabilities in a custom apply_func",
    "VotingRegressor": "no direct skrub primitive: replicate each base estimator via "
                        ".skb.apply, average predictions in a custom apply_func",
}


def _expand_ensemble_into_steps(est, est_cls: str, steps: list, notes: list) -> None:
    """Expand StackingRegressor/Classifier/VotingRegressor/Classifier nested estimators.

    _params_of() deliberately drops non-JSON-primitive values, which silently
    swallows the `estimators` list and `final_estimator` of ensemble wrappers.
    This helper recovers them as separate StepIR entries so the converter sees
    every sub-model explicitly.
    """
    try:
        from sklearn.ensemble import (StackingClassifier, StackingRegressor,
                                      VotingClassifier, VotingRegressor)
    except ImportError:
        return

    stacking_types = (StackingRegressor, StackingClassifier)
    voting_types = (VotingRegressor, VotingClassifier)
    if not isinstance(est, stacking_types + voting_types):
        return

    notes.append(
        f"{est_cls} has nested estimators with no single skrub primitive equivalent. "
        "Each base estimator should be wired as a separate .skb.apply(..., y=y) branch; "
        "predictions must be wrapped as DataFrames and concatenated with .skb.concat() "
        "before the meta-learner step."
    )

    for _, base_est in est.estimators:
        base_cls = type(base_est).__name__
        steps.append(StepIR(
            kind="estimator",
            library=type(base_est).__module__.split(".")[0],
            class_or_func=base_cls,
            params=_params_of(base_est),
            applies_to=f"{est_cls}.base_estimator",
            skrub_hint=SKRUB_HINTS.get(base_cls, "apply via .skb.apply(estimator, y=y)"),
        ))

    if isinstance(est, stacking_types):
        fe = est.final_estimator
        fe_cls = type(fe).__name__
        steps.append(StepIR(
            kind="estimator",
            library=type(fe).__module__.split(".")[0],
            class_or_func=fe_cls,
            params=_params_of(fe),
            applies_to=f"{est_cls}.final_estimator",
            skrub_hint=SKRUB_HINTS.get(fe_cls, "apply via .skb.apply(estimator, y=y)"),
        ))


def _params_of(estimator) -> dict[str, Any]:
    try:
        params = estimator.get_params(deep=False)
    except Exception:
        return {}
    out = {}
    for k, v in params.items():
        # keep only JSON-ish values; complex nested estimators get their own StepIR
        if isinstance(v, (int, float, str, bool)) or v is None:
            out[k] = v
    return out


def analyze_estimator(
    pipeline,
    feature_columns: list[str],
    target_column: str,
    task: str,
    df_dtypes: Optional[dict[str, str]] = None,
) -> PipelineIR:
    """Introspect a live sklearn Pipeline/ColumnTransformer + final estimator."""
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline

    steps: list[StepIR] = []
    column_groups: list[ColumnGroup] = []
    notes: list[str] = []

    def classify_cols(cols: list[str]) -> str:
        if not df_dtypes:
            return "unknown"
        kinds = {df_dtypes.get(c, "unknown") for c in cols}
        if kinds <= {"int64", "float64"}:
            return "numeric"
        if kinds <= {"object", "category"}:
            return "categorical"
        return "mixed"

    def walk(step, name_prefix=""):
        if isinstance(step, Pipeline):
            for sname, sub in step.steps:
                walk(sub, f"{name_prefix}{sname}.")
        elif isinstance(step, ColumnTransformer):
            for sname, sub, cols in step.transformers:
                if sub in ("drop", "passthrough"):
                    continue
                cols = list(cols) if not isinstance(cols, str) else [cols]
                group = ColumnGroup(name=f"{name_prefix}{sname}", columns=cols,
                                     dtype_hint=classify_cols(cols))
                column_groups.append(group)
                start_idx = len(steps)
                walk(sub, f"{name_prefix}{sname}.")
                for s in steps[start_idx:]:
                    s.applies_to = group.name
        else:
            cls = type(step).__name__
            steps.append(StepIR(
                kind="transformer",
                library=type(step).__module__.split(".")[0],
                class_or_func=cls,
                params=_params_of(step),
                skrub_hint=SKRUB_HINTS.get(cls),
            ))

    if isinstance(pipeline, Pipeline):
        *pre_steps, (final_name, final_est) = pipeline.steps
        for sname, sub in pre_steps:
            walk(sub, f"{sname}.")
        est_cls = type(final_est).__name__
        estimator = StepIR(
            kind="estimator",
            library=type(final_est).__module__.split(".")[0],
            class_or_func=est_cls,
            params=_params_of(final_est),
            skrub_hint=SKRUB_HINTS.get(est_cls, "apply via .skb.apply(estimator, y=y)"),
        )
        _expand_ensemble_into_steps(final_est, est_cls, steps, notes)
    else:
        walk(pipeline)
        estimator = steps.pop() if steps and steps[-1].kind != "transformer" else StepIR(
            kind="estimator", library="unknown", class_or_func="unknown")

    if not column_groups:
        notes.append(
            "No ColumnTransformer found — pipeline applies uniformly to all "
            "feature columns. Converter should likely use a single "
            "skrub.TableVectorizer() over the whole frame."
        )

    table = TableIR(var_name="df", role="primary", columns=feature_columns + [target_column])

    return PipelineIR(
        task=task,
        tables=[table],
        column_groups=column_groups,
        steps=steps,
        target_column=target_column,
        target_table="df",
        estimator=estimator,
        notes=notes,
    )


class _CustomCodeVisitor(ast.NodeVisitor):
    """Very lightweight AST pass: pulls out top-level function defs and
    pandas-ish method chains (merge/groupby/agg) as candidate StepIR/TableIR
    entries with source captured verbatim, for imperative (non-Pipeline)
    scripts. This is intentionally shallow — it is a *net* to make sure the
    converter sees the custom logic, not a full pandas semantic analyzer."""

    PANDAS_SIGNAL_CALLS = {"merge", "groupby", "agg", "pivot_table", "join", "apply", "map"}

    def __init__(self, source: str):
        self.source = source
        self.custom_steps: list[StepIR] = []
        self.tables: list[TableIR] = []

    def visit_FunctionDef(self, node: ast.FunctionDef):
        snippet = ast.get_source_segment(self.source, node)
        self.custom_steps.append(StepIR(
            kind="custom_function",
            library="pandas",
            class_or_func=node.name,
            source_snippet=snippet,
        ))
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        if isinstance(node.func, ast.Attribute) and node.func.attr in self.PANDAS_SIGNAL_CALLS:
            snippet = ast.get_source_segment(self.source, node)
            kind = "groupby_agg" if node.func.attr in ("groupby", "agg", "pivot_table") else \
                   "join" if node.func.attr in ("merge", "join") else "custom_function"
            self.custom_steps.append(StepIR(
                kind=kind, library="pandas", class_or_func=node.func.attr,
                source_snippet=snippet,
            ))
        self.generic_visit(node)


def _ast_target_column(tree: ast.AST, default: str = "target") -> str:
    """Scan for `TARGET_COLUMN = "..."` or `target_col = "..."` assignments."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if (isinstance(tgt, ast.Name)
                        and tgt.id.upper() in ("TARGET_COLUMN", "TARGET_COL")):
                    if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                        return node.value.value
    return default


def analyze_source(path: str, task: str = "classification",
                    target_column: Optional[str] = None) -> PipelineIR:
    """AST fallback for scripts without an explicit sklearn Pipeline object."""
    with open(path) as f:
        source = f.read()
    tree = ast.parse(source)
    visitor = _CustomCodeVisitor(source)
    visitor.visit(tree)

    if target_column is None:
        target_column = _ast_target_column(tree, default="target")

    notes = [
        "Analyzed via AST fallback (no sklearn Pipeline/ColumnTransformer object "
        "detected). All groupby/merge/custom-function steps below must be "
        "re-implemented by the converter as skrub DataOps nodes acting on "
        "skrub.var(...)-derived DataOps, not as raw pandas on the source df."
    ]
    return PipelineIR(
        task=task,
        tables=[TableIR(var_name="df", role="primary", columns=[])],
        column_groups=[],
        steps=visitor.custom_steps,
        target_column=target_column,
        target_table="df",
        estimator=StepIR(kind="estimator", library="unknown", class_or_func="unknown"),
        notes=notes,
    )
