# Skrubifier — Phase 2 Evaluation Table (LLM-Generated Conversions)

## Setup

- **Model**: `qwen3-coder-next` (GWDG/AcademicCloud SAIA endpoint)
  — The originally configured default `qwen3-coder-30b-a3b-instruct` was not
  available; `qwen3-coder-next` was chosen as the closest available coding-capable
  model. The default was updated in `converter.py` and `cli.py`.
- **Backend**: GWDG (`--backend gwdg`)
- **Max repair rounds**: 3 (default)
- **Static check**: same `validator.py` `static_check()` used in Phase 1
- **Dynamic check**: adapted harnesses run each LLM script in a subprocess,
  pointing `learner.fit({"<llm_var_name>": data})` at the LLM script's actual
  `skrub.var(...)` key rather than the hand-written reference's key.
  Tolerance: `max(0.02, 0.03 * |original_metric|)`.

---

## Results

| # | Name | Static OK | Attempts | Dynamic OK | Error / Root Cause |
|---|------|-----------|----------|------------|--------------------|
| 01 | Titanic | Pass | 1 | Fail | `df.drop(columns=[...])` on `skrub.var("df", None)` — pandas `.drop()` instead of `.skb.drop()`; preview evaluation fails at script load (NoneType has no attribute 'drop') |
| 02 | House Prices XGBoost | Pass | 1 | Fail | `import pandas as pd` missing — `pd.read_csv(...)` used in `skrub.var()` default but not imported; NameError at script load |
| 03 | Credit Fraud Multitable | Pass | 1 | Fail | `AggJoiner(aux_table=products_df)` where `products_df` is a DataOp (`skrub.var("products", None)`) — AggJoiner requires a real DataFrame for aux_table, not a DataOp |
| 04 | NYC Taxi Fare | Pass | 1 | Fail | `skrub.StandardScaler()` — StandardScaler is `sklearn.preprocessing.StandardScaler`, not a skrub class |
| 05 | Otto Group | Pass | 1 | Fail | `df.skb.select(...)` on `skrub.var("df", None)` placeholder — preview evaluation fails (NoneType); also `X.skb.mark_as_X()` return value discarded |
| 06 | Allstate Claims Severity | Pass | 1 | Fail | `pd.concat([DataOp, DataOp], axis=1)` — pandas `concat` does not accept DataOp objects; should use `.skb.concat()` |
| 07 | Random Acts of Pizza | Pass | 1 | Fail | `df.drop(columns=[...])` on `skrub.var("df", None)` — same as example 01; pandas `.drop()` fails on NoneType preview |
| 08 | Spooky Author | Pass | 1 | Fail | `skrub.TextEncoder(encoding=..., lowercase=..., max_features=..., ...)` — `TextEncoder` does not accept TF-IDF parameters; wrong API usage |
| 09 | Home Credit | Pass | 1 | Fail | `app_df = skrub.var("app", app_df)` — `app_df` referenced before assignment (circular/forward reference); NameError at script load |
| 10 | Santander | Pass | 1 | Fail | `df.drop(columns=["target"])` on `skrub.var("df", None)` — same as examples 01, 07; pandas `.drop()` fails on NoneType preview |

---

## Summary

| Metric | Count |
|--------|-------|
| Pipelines run | 10/10 |
| Static check pass | 10/10 |
| Dynamic check pass | **0/10** |
| Average repair rounds | 1.0 (no repairs needed — all static passes on first attempt) |

---

## Structural Comparison Notes (LLM vs Hand-Written Reference)

| # | LLM Approach | Reference Approach | Match? |
|---|-------------|-------------------|--------|
| 01 | `df.drop(columns=[target])` to create X (all non-target cols) | Explicit 7-column selection | Partial — different column set |
| 02 | `apply_func(np.log1p)` on target before `mark_as_y()` (simpler) | `TransformedTargetRegressor(func=log1p, inverse_func=expm1)` to keep log transform inside estimator | Different — LLM approach is simpler but predictions are in log scale |
| 03 | `AggJoiner` (cached aggregations, known to cause NaN for test entities) | `apply_func(_agg_products)` + DataOp `.merge()` (stateless, re-evaluated at each call) | Structurally different — LLM picked the anti-pattern the reference explicitly rejected |
| 04 | Helper functions via `apply_func`, `skrub.eval_mode()` for train-only filter | Same pattern overall, but uses `skrub.StandardScaler()` (hallucinated) | Similar structure, key API error |
| 05 | Direct `CalibratedClassifierCV` with fixed params (no hyperparameter search) | `choose_from()` grid search over representative param grid | Different — LLM omits the `choose_from` API entirely |
| 06 | Three `.skb.apply()` branches + `pd.concat()` (wrong) for stacking | Three `.skb.apply()` branches + `.skb.concat()` (correct) | Similar intent, wrong concat method |
| 07 | ColumnTransformer replicated inside `.skb.apply(ColumnTransformer(...))` | Single `TableVectorizer()` (auto-handles text + numeric columns) | Different — LLM replicated the source sklearn structure instead of using TableVectorizer |
| 08 | `TableVectorizer(text_transformer=TextEncoder(invalid_params))` | `apply_func(lambda df: df["text"])` + `.skb.apply(TfidfVectorizer(...))` directly | Different approach, wrong API params |
| 09 | `AggJoiner` + circular variable reference | `apply_func(_agg_bureau)` + DataOp `.merge()` | Different — same anti-pattern as 03, plus a NameError |
| 10 | Three separate `.skb.apply()` calls (Scaler, SelectKBest, LR) | Single `sklearn.Pipeline` wrapped in one `.skb.apply()` | Different — both are valid; LLM's version would work if `df.drop()` didn't fail |

---

## Root Cause Analysis

The 0/10 dynamic pass rate despite 10/10 static pass rate reveals that the static checker
is insufficient as the sole quality gate. The systematic errors across all 10 scripts fall
into four categories:

1. **`None` placeholder anti-pattern** (01, 05, 07, 10 — 4 failures):
   The LLM consistently uses `skrub.var("name", None)` as a placeholder. This causes all
   subsequent DataOps operations to fail during skrub's preview-evaluation phase (which
   runs at script load time using the default value). The correct pattern is to either:
   (a) provide a real DataFrame sample, or (b) use `.skb.drop()` / `.skb.select()` instead
   of pandas methods, which defer evaluation. The prompt's `SKRUB_API_REFERENCE` apparently
   does not make this distinction clear.

2. **Missing imports** (02 — 1 failure):
   The LLM uses `pd.read_csv(...)` without importing pandas. A simple static import check
   would catch this; the current `static_check()` does not.

3. **Sklearn/skrub namespace confusion** (04, 08 — 2 failures):
   `skrub.StandardScaler` and `skrub.TextEncoder(encoding=..., max_features=...)` — the
   LLM hallucinates skrub wrappers for sklearn classes that don't exist in skrub, or passes
   sklearn-style params to skrub classes that have different signatures.

4. **DataOps API misuse** (03, 06, 09 — 3 failures):
   - `AggJoiner(aux_table=DataOp)` — AggJoiner needs a real DataFrame, not a DataOp
   - `pd.concat([DataOp, ...])` — must use `.skb.concat()`
   - `skrub.var("name", undefined_var)` — circular reference

---

## Phase 1 vs Phase 2 Comparison

| | Phase 1 (hand-written) | Phase 2 (LLM-generated) |
|---|----------------------|------------------------|
| Static pass rate | 10/10 (100%) | 10/10 (100%) |
| Dynamic pass rate | 9/10 (90%) | **0/10 (0%)** |
| Repair rounds needed | n/a (hand-written) | 0 (static only, no dynamic repair) |

The 90-point gap in dynamic pass rate is the key finding of Phase 2. Static validation
(Python syntax + import checks) is necessary but far from sufficient: it catches only the
most basic syntactic errors, while leaving all semantic DataOps API errors undetected. The
automated pipeline as currently designed would need a stronger static check (e.g., import
resolution, attribute-existence checks against the actual skrub module, None-placeholder
detection) or a runtime preview step before the "static_ok=True" verdict can be trusted.

The repair loop (up to 3 rounds) was never triggered because all scripts passed static
validation immediately — the errors are semantic, not syntactic, so the repair signal
(static_check feedback) never fires.
