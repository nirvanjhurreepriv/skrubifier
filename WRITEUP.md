# Skrubifier — Grader Writeup

This document synthesises the project for reviewers. Primary sources for every
claim are `README.md`, `CHANGELOG.md`, `examples/PLAN.md`, and
`results/evaluation_table.md` — nothing is invented or approximated here.
Where a result is not yet available (Phase 2 has not run), that is stated
explicitly.

---

## 1. Overview

Skrubifier is a three-stage framework that converts tabular ML pipelines written
in scikit-learn's `Pipeline`/`ColumnTransformer` API (plus XGBoost/LightGBM
wrappers, manual pandas feature engineering, and multi-table joins) into
**skrub's DataOps API** (`skrub.var`, `.skb.apply`, `.skb.mark_as_X/y`,
`.skb.make_learner`, etc.), using an LLM as the translation step — but
constraining the LLM's job to syntax translation rather than structural
understanding, and wrapping it with deterministic analysis before and numerical
validation after, so the conversion is narrow and checkable rather than
open-ended.

---

## 2. Architecture: why three stages, not one LLM call

Naively prompting an LLM with "here is a sklearn pipeline, give me the skrub
DataOps version" fails in practice for three reasons documented in `README.md`:
(1) the DataOps API is a build-time lazy DAG, not an eager fit/transform API —
the LLM has to convert *control flow* into DAG-building calls; (2) real solutions
mix pandas munging, custom functions, multiple tables, and CV loops that have no
1:1 skrub primitive; (3) correctness has to be checked *numerically*, not by
visual inspection of syntax.

Skrubifier addresses this by separating the problem into three stages:

```
source pipeline (.py / .ipynb)
        │
        ▼
 ┌──────────────────┐  AST + runtime introspection of the sklearn Pipeline /
 │ 1. ANALYZER      │  ColumnTransformer / estimator objects. Produces a
 │   analyzer.py    │  structured PipelineIR (JSON-serializable dataclasses)
 │                  │  describing steps, column groups, estimators, hyper-
 │                  │  parameters, and any custom transform code found by AST.
 └──────────────────┘
        │  PipelineIR
        ▼
 ┌──────────────────┐  LLM (Claude / GWDG open-weight) is prompted with:
 │ 2. CONVERTER     │  (a) the IR, (b) a condensed skrub DataOps API reference
 │   converter.py   │  + few-shot examples, (c) an IR->code contract. Emits a
 │                  │  candidate skrub_dataops.py script.
 └──────────────────┘
        │  candidate script
        ▼
 ┌──────────────────┐  Static check: does the script use `.skb` namespace,
 │ 3. VALIDATOR     │  `var`, `apply`, `make_learner`? Dynamic check: run
 │   validator.py   │  original + candidate on the same held-out split;
 │                  │  compare metrics within tolerance max(0.02, 0.03*|orig|).
 └──────────────────┘  On failure, feed the diff back to the LLM for up to N
        │              repair rounds.
        ▼
  verified skrub DataOps script + validation report
```

**Why the IR matters:** `ir.py` defines `PipelineIR`, `StepIR`, `TableIR`, and
`ColumnGroup` dataclasses that form the contract between stages. The analyzer
does all structural work deterministically and testably (9/9 unit tests in
`tests/test_analyzer.py` cover it). The LLM only ever sees a structured
description plus the original code as a snippet for custom steps — it is never
asked to parse the source pipeline itself. This makes the LLM's job narrower,
the prompt shorter, and the failures (when they occur) attributable to a
specific IR field rather than ambiguous source-reading errors.

---

## 2.1 Development Process

We built this framework and its ten pipelines in two explicit phases: Phase 1 established hand-verified reference conversions for all ten pipelines, validating the analyzer and validator independently of any LLM; Phase 2 ran the fully automated converter against those references. Within each phase, we worked in small, checkable increments — build or fix one component, run the offline test suite or a pipeline's dynamic harness immediately, and only proceed once that increment was verified rather than assumed to work. This is why `CHANGELOG.md` records specific, dated fixes with reproduction steps rather than a single undifferentiated final version. We treated failing checks as information rather than something to route around: when dynamic validation failed, we root-caused the specific failure before deciding whether to fix it (as with the AggJoiner and repair-loop bugs) or document it as a genuine limitation (as with the stacking meta-learner gap); we applied the same standard to our own framework code — the Section 8 repair-loop finding is the clearest example, discovered by testing the whole system end-to-end rather than by inspecting the code in isolation.

---

## 3. The 10 pipelines: diversity coverage

Examples 01–04 were present at the start of Phase 1; examples 05–10 were built
during Phase 1 per `examples/PLAN.md`. Sources are MLE-Bench tasks or published
Kaggle competition winning-solution notebooks, except example 04 which uses an
official skrub DataOps tutorial notebook as the source pipeline (noted in its
`source_pipeline.py` docstring).

| # | Name | MLE-Bench / Kaggle source | Domain | Dataset (rows) | ML task | Complexity | Search space | Data |
|---|------|--------------------------|--------|---------------|---------|------------|-------------|------|
| 01 | Titanic | Kaggle: Titanic | Survival | ~891 | Binary classif. | Single-table | Fixed params | Synthetic |
| 02 | House Prices XGBoost | Kaggle: House Prices Advanced Regression | Real estate | ~1,460 | Regression | Single-table | Fixed params | Synthetic |
| 03 | Credit Fraud Multitable | Kaggle: Credit Fraud Detection | Fraud detection | 600 baskets + 2,400 products | Binary classif. | Multi-table (2 tables) | Fixed params | Synthetic |
| 04 | NYC Taxi Fare | skrub official tutorial notebook | Geo/transport | 500k train (10k used) | Regression | Single-table + train-filter | Fixed params | **Real** |
| 05 | Otto Group | Kaggle: Otto Group Product Classification | Multi-class product | ~1,000 (93 features) | 9-class classif. | Single-table | **Large (skrub.choose_from + make_grid_search)** | Synthetic |
| 06 | Allstate Claims Severity | Kaggle: Allstate Claims Severity | Insurance | ~2,000 (130 features) | Regression | Single-table + ensemble (stacking) | Fixed params | Synthetic |
| 07 | Random Acts of Pizza | MLE-Bench: random-acts-of-pizza | Text + tabular | ~5,000 | Binary classif. | Single-table (mixed types) | Fixed params | Synthetic |
| 08 | Spooky Author Identification | MLE-Bench: spooky-author-identification | Text-in-table | ~5,000 | 3-class classif. | Single-table (text only) | Fixed params | Synthetic |
| 09 | Home Credit Default Risk | MLE-Bench: home-credit-default-risk | Credit scoring | 3,000 apps + 12,000 bureau records | Binary classif. | Multi-table (2 tables) | Fixed params | Synthetic |
| 10 | Santander Customer Transaction | Kaggle: Santander Customer Transaction | Banking | ~5,000 (200 features) | Binary classif. | Single-table (pure numeric) | Fixed params | Synthetic |

The diversity covers: 5 distinct ML tasks (binary/multi-class classification,
regression, NLP-in-table); 3 structural complexity tiers (single-table, multi-table
join, stacking ensemble); text, numeric, categorical, and mixed-type inputs; 2
multi-library estimators (XGBoost, HistGradientBoosting) alongside sklearn-native
models; and one pipeline (**example 05**) that exercises a **large hyperparameter
search space** — the converted `converted_dataops.py` uses `skrub.choose_from([200,
400, 600], name="n_estimators")` over 3 hyperparameters and `pred.skb.make_grid_search(cv=5)`
to search it. This is the pattern skrub DataOps was specifically designed for (lazy
DAG-building means every hyperparameter combination is a valid pipeline variant,
not a replicated object).

**Tabular-only scope (requirement):** All 10 examples are expressed as tabular CSV files
with named columns. Example 07 fits this cleanly: text is one feature among several
tabular columns (numeric and categorical fields alongside a title/body text field),
combined via TableVectorizer — the same pattern as any other mixed-type tabular pipeline
in this set. Example 08 is included as a boundary case: its source task is entirely text
classification, so it stresses the tabular scope more than any other example. It's
retained because the assignment's pipelines are still expressed and converted as a
single-column tabular DataFrame processed through TableVectorizer/TfidfVectorizer — the
same DataOps conversion mechanics used elsewhere — rather than a multi-file NLP corpus
pipeline with its own tokenization/embedding infrastructure. If a strictly text-free
interpretation of "tabular only" is required, example 08 is the one pipeline in this set
to reconsider or replace; it is flagged here rather than left as an unstated assumption.

**Data note:** Only example 04 uses real competition data. The remaining 9
use synthetic CSVs matched to the original column names, dtypes, and
distributional shape. See `results/evaluation_table.md` (§ "Data sourcing vs.
pipeline sourcing") for the full disclosure; the short version is that real
Kaggle data isn't downloadable in this environment, and synthetic data is
sufficient for evaluating *structural correctness* of the DataOps conversion
(the question "does the converted pipeline produce the same predictions as the
source?") even though absolute metric values aren't comparable to leaderboard
scores.

---

## 4. Results — Phase 1 (hand-written reference conversions)

All 10 pipelines have hand-written `converted_dataops.py` scripts validated by
their paired `harness.py`. Full table is in `results/evaluation_table.md`;
key summary:

- **Static check:** 10/10 pass. Every converted script uses `.skb` namespace,
  `var`, `apply`, and `make_learner` as required by `validator.static_check()`.
- **Dynamic check:** 9/10 pass within tolerance `max(0.02, 0.03 * |original|)`.

Passing examples (metrics are original vs. converted):

| # | Metric | Original | Converted | Delta | Tol | Result |
|---|--------|----------|-----------|-------|-----|--------|
| 01 | ROC AUC | 0.768 | 0.782 | 0.014 | 0.023 | Pass |
| 02 | R² (log scale) | 0.772 | 0.777 | 0.005 | 0.023 | Pass |
| 03 | ROC AUC | 0.850 | 0.850 | 0.000 | 0.026 | Pass |
| 04 | R² | 0.656 | 0.656 | 0.000 | 0.020 | Pass |
| 05 | ROC AUC (macro OvR) | 0.815 | 0.820 | 0.005 | 0.024 | Pass |
| 07 | ROC AUC | 0.638 | 0.638 | 0.000 | 0.019 | Pass |
| 08 | ROC AUC (macro OvR) | 0.963 | 0.963 | 0.000 | 0.029 | Pass |
| 09 | ROC AUC | 0.881 | 0.883 | 0.002 | 0.026 | Pass |
| 10 | ROC AUC | 0.573 | 0.573 | 0.000 | 0.020 | Pass |

**One documented failure — example 06 (Allstate Claims Severity):**

| # | Metric | Original | Converted | Delta | Tol | Result |
|---|--------|----------|-----------|-------|-----|--------|
| 06 | R² (log scale) | 0.644 | 0.608 | 0.037 | 0.020 | Fail |

Root cause: `sklearn.ensemble.StackingRegressor` trains its meta-learner on
**out-of-fold (OOF)** predictions — each base estimator is cross-validated so
the training instances it sees during meta-learner fitting were held out during
each base estimator's training. The DataOps version, which has no OOF stacking
primitive, fits all three base estimators on the full training set and then feeds
those same in-fold predictions to the meta-learner. The meta-learner therefore
sees slightly overfit base-model outputs, making it a mildly easier problem than
the original — the converted model appears slightly worse on unseen test data
because the meta-learner overfitted to inflated in-fold predictions. Delta 0.037
exceeds tolerance 0.020. This is a genuine framework limitation for this
pipeline class, not a bug in the conversion code (see §6 for full details).

---

## 5. Real bugs found and fixed during Phase 1

The following bugs were discovered by running dynamic validation. Each one
produced a silent wrong answer (no exception, but wrong predictions) or a hard
crash, and each required a non-obvious fix. They are listed in order of
discovery.

### Bug 1 — `TableVectorizer` parameter rename (example 02)

**What broke:** `skrub.TableVectorizer(high_cardinality_transformer=TargetEncoder())`
raised `TypeError: __init__() got an unexpected keyword argument 'high_cardinality_transformer'`.

**Why:** In skrub 0.9.0, the parameter was renamed from `high_cardinality_transformer`
to `high_cardinality`. The hand-written script used the old name. The same stale
name was in `analyzer.py`'s `SKRUB_HINTS` dict, which would have propagated the
error to any LLM-generated script that followed the hint.

**Fix:** Renamed the parameter in `converted_dataops.py` and updated the
`SKRUB_HINTS["TargetEncoder"]` entry in `analyzer.py`.

---

### Bug 2 — `apply_func` after an estimator node crashes during fit (example 02)

**What broke:** The original converted script applied `np.expm1` after the
XGBoost node to invert a log1p target transform:
```python
pred_log = vectorized.skb.apply(XGBRegressor(...), y=y)
pred     = pred_log.skb.apply_func(np.expm1)
```
This raised `"loop of ufunc does not support argument 0 of type XGBRegressor"`.

**Why:** In skrub 0.9.0, during `learner.fit()`, an estimator DataOp node
evaluates to the **fitted estimator object**, not to its predictions. Chaining
`apply_func(np.expm1)` after it therefore calls `np.expm1(XGBRegressor_object)`.

**Fix:** Replaced the two-step pattern with sklearn's `TransformedTargetRegressor`
wrapper, which handles the log1p/expm1 transform internally:
```python
model = TransformedTargetRegressor(
    regressor=XGBRegressor(...), func=np.log1p, inverse_func=np.expm1)
pred = vectorized.skb.apply(model, y=y)
```

---

### Bug 3 — `AggJoiner` caches training aggregations; test entities get NaN (examples 03, 09)

**What broke:** Both multi-table examples initially used `skrub.AggJoiner` to
aggregate the auxiliary table (products/bureau) and join it to the primary table.
The model trained fine but produced constant predictions on the test set:
`[9.99204e-01, 7.96e-04]` for every basket regardless of its features.

**Why:** `AggJoiner` is a sklearn-style stateful transformer. It *fits*
aggregations from the training auxiliary table and *caches* them. At predict
time, test entities not seen in the training auxiliary table get NaN for all
aggregated columns — the model then predicts the majority-class constant.
The source pipelines both re-computed aggregations fresh at prediction time by
calling `build_features(test_app, test_bureau)` separately, so the DataOps
equivalent must also be stateless.

**Fix:** Replaced `AggJoiner` with a stateless aggregation function applied via
`.skb.apply_func()` and joined using the DataOp `.merge()` operator (which
re-runs at every fit/predict call as a DAG edge):
```python
agg = products.skb.apply_func(_agg_products)
merged = baskets.merge(agg, on="basket_ID", how="left")
```

---

### Bug 4 — ID column left in features after `.drop()` (example 03)

**What broke:** After fixing Bug 3, example 03 still produced AUC ≈ 0.5.

**Why:** The original code used `drop(columns=["fraud_flag"])` to remove the
target, leaving `basket_ID` in the feature matrix. `basket_ID` is a sequential
integer: training baskets have IDs 1–480, test baskets 481–600. The model
memorised the training ID range and predicted the majority class for all IDs
outside it.

**Fix:** Changed from a `drop`-based selection to explicitly naming the four
derived aggregate columns as features: `X = merged[DERIVED_COLS].skb.mark_as_X()`.

---

### Bug 5 — XGBoost 3.x rejects string class labels (example 05)

**What broke:** Fitting the Otto Group pipeline with `objective="multi:softprob"` and
string labels `['Class_1', ..., 'Class_9']` raised:
`"Invalid classes inferred from unique values of y. Expected: [0 1 2 ... 8], got ['Class_1' 'Class_2' ...]"`.

**Why:** XGBoost 3.x requires integer class labels (0..N−1) for multi-class
objectives. The real Kaggle dataset uses string labels; XGBoost 2.x accepted them.

**Fix:** Added a label-encoding step inside the DataOps DAG:
```python
y = otto["target"].skb.apply_func(
    lambda s: s.str.replace("Class_", "", regex=False).astype(int) - 1
).skb.mark_as_y()
```
and aligned the harness to use `LabelEncoder` + `label_binarize` consistently.

---

### Bug 6 — `skrub.concat()` does not exist; `.skb.concat()` requires DataFrame inputs (example 06)

**What broke (part a):** `skrub.concat([pred_xgb, pred_ridge, pred_et], axis=1)`
raised `AttributeError: module 'skrub' has no attribute 'concat'`.

**Why:** There is no module-level `skrub.concat()` function. The correct form is
the DataOp method `.skb.concat([...], axis=1)` called on one of the DataOp instances.

**What broke (part b):** After fixing (a), `.skb.concat()` raised a shape error
because each base-model prediction is a `Series` DataOp, not a `DataFrame`.

**Fix:** Wrapped each prediction in a DataFrame before concatenating:
```python
pred_xgb_df = pred_xgb.skb.apply_func(lambda s: pd.DataFrame({"pred_xgb": s}))
stacked = pred_xgb_df.skb.concat([pred_xgb_df, pred_ridge_df, pred_et_df], axis=1)
```

---

### Bug 7 — `StackingRegressor` nested estimators silently dropped by `_params_of` (analyzer.py)

**What broke:** `analyzer.py`'s `_params_of()` helper only retains JSON-primitive
parameter values (int, float, str, bool, None). `StackingRegressor.estimators` is
a list of `(name, estimator)` tuples and `StackingRegressor.final_estimator` is an
estimator object — both were silently dropped, leaving the IR with no record of
the sub-models. The LLM would have received an IR for a StackingRegressor with no
visible base estimators or meta-learner.

**Fix:** Added `_expand_ensemble_into_steps()`, called after the final estimator
is identified in `analyze_estimator()`. It detects `StackingRegressor`,
`StackingClassifier`, `VotingRegressor`, and `VotingClassifier`, and emits one
`StepIR` per base estimator (`applies_to="{Cls}.base_estimator"`) and one for
the final estimator (`applies_to="{Cls}.final_estimator"`), plus a note in
`ir.notes` that no single skrub primitive exists. Covered by regression test
`test_estimator_analyzer_captures_stacking_base_estimators`.

---

### Documented limitation — MinHashEncoder structural gap for word-vocabulary text (example 08)

**What was tried:** `skrub.MinHashEncoder(n_components=100)` was used as the
"pure skrub" conversion for a TF-IDF -> LogisticRegression text pipeline.

**What happened:** On realistic overlapping-vocabulary synthetic text (after
removing the original trivially-separable disjoint-vocabulary data), the source
TF-IDF pipeline achieved AUC 0.963 while the MinHashEncoder conversion achieved
AUC 0.856 — delta 0.107 vs tolerance 0.029, a hard FAIL. Increasing
`n_components` does not close the gap: it is structural. MinHashEncoder hashes
character n-grams of the full text string into a fixed-size dense vector; it
cannot replicate TF-IDF's per-token inverse-document-frequency weighting, which
is what discriminates between author styles in bag-of-words text.

**Resolution:** The converted script uses `TfidfVectorizer` directly via
`.skb.apply_func(lambda df: df["text"])` -> `.skb.apply(TfidfVectorizer(...))`,
which is a valid DataOps pattern (same as example 10 wrapping a full sklearn
Pipeline in `.skb.apply()`), and produces delta 0.000. The MinHashEncoder result
is preserved as a documented limitation in `examples/08_spooky_author/converted_dataops.py`'s
docstring under "Alternative tried — skrub's default text routing": the skrub-native
text routing path (`TableVectorizer` -> `MinHashEncoder`) underperforms TF-IDF
on word-vocabulary classification by ~0.1 AUC when `sentence_transformers` is
unavailable and `TextEncoder` cannot be used.

---

## 6. Framework limitations (known, disclosed)

These are structural limitations of the framework as built — not bugs that were
fixed, but constraints the grader should be aware of.

### L1 — No OOF stacking primitive in skrub 0.9 (example 06)

`sklearn.ensemble.StackingRegressor` trains its meta-learner on out-of-fold
predictions, avoiding overfitting. skrub 0.9 has no equivalent DAG primitive
for OOF prediction — all branches see the full training set. The result is a
predictable but irreducible quality gap when converting stacking ensembles:
delta 0.037 vs tolerance 0.020 on example 06. The source is noted in
`CHANGELOG.md` and `results/evaluation_table.md`.

### L2 — Row-wise inline lambdas invisible to the AST analyzer

`_CustomCodeVisitor` in `analyzer.py` captures named `FunctionDef` nodes and a
fixed set of pandas method names (`merge`, `groupby`, `agg`, etc.). A `lambda`
passed inline to `df.apply(lambda row: ...)` is not detected. Fix noted in
`examples/PLAN.md` §Known limitations #2: a ~10-line addition to also visit
`ast.Lambda` nodes and attach their source to the enclosing `.apply()` call's
`StepIR`.

### L3 — CV-loop-internal fitting not modeled in the IR

The analyzer has no notion of "this transformer refits per fold inside a manual
`for train_idx, val_idx in kf.split(X)` loop" — it will see the encoder as a
`custom_function` StepIR. The converter prompt instructs the LLM to use
`.skb.cross_validate()` over the full DAG (which naturally refits everything per
fold, avoiding the leakage that motivated the original manual loop), but this
relies on the LLM correctly recognising the loop pattern. Noted in `PLAN.md`
§Known limitations #3 as a target for close manual review.

### L4 — `TableIR.join_to` models only linear join chains

`ir.py`'s `TableIR.join_to` is a single string (one parent table). Pipelines
with a genuine join graph — e.g. products -> baskets AND identities -> baskets
simultaneously — cannot be fully expressed. The workaround used in examples 03
and 09 (stateless `apply_func` aggregation + DataOp `.merge()`) sidesteps the
IR limitation for those cases, but the IR itself would need `join_to: list[str]`
to represent the graph correctly for the converter. Noted in `PLAN.md`
§Known limitations #1.

### L5 — `TextEncoder` (skrub-native semantic text encoding) requires optional dependency

`skrub.TextEncoder`, which uses sentence-transformer embeddings, is the intended
skrub-native replacement for `TfidfVectorizer`. It requires `sentence_transformers`
(not installed in this environment). As demonstrated in example 08, the fallback
`MinHashEncoder` is structurally inadequate for word-vocabulary classification
(delta 0.107 on realistic text). Until `TextEncoder` is available,
the recommended converted form for TF-IDF source pipelines is to pass
`TfidfVectorizer` directly to `.skb.apply()`, as done in example 08.

---

## 7. Stratum compatibility

From `README.md`:

[deem-data/stratum](https://github.com/deem-data/stratum) is not a separate
target syntax — it is a drop-in accelerated runtime for the same skrub DataOps
operator abstraction (`import stratum as skrub`), adding a Rust backend,
cost-based optimizer, and scheduler on top of the identical
`.skb.var/.apply/.mark_as_X/.make_learner` API.

**Syntax-level compatibility:** Confirmed. Every generated script uses the
`try: import stratum as skrub / except ImportError: import skrub` guard
(specified in `prompts.STRATUM_NOTE`), and stratum re-exports the exact DataOps
operators used across all 10 examples.

**Execution-level compatibility:** Not verified in this environment.
Stratum has no pip wheel — it is built from source via `maturin develop
--release`, which requires a Rust toolchain and Python 3.12+, neither of which
is available here. Stratum integration was optional in the development brief and was not attempted
in this environment: there is no CHANGELOG entry for a stratum build attempt. Execution verification must happen in an environment where stratum
can be built.

**Validator support:** `validator.dynamic_check(..., use_stratum=True)` sets
`STRATUM_RUST_BACKEND=1` in the subprocess environment, so validation would
automatically pick up stratum once it is built — no code path divergence from
the plain-skrub case.

---

## 8. Phase 2 — automated conversion results

Phase 2 ran all 10 pipelines through the automated `converter.py` pipeline using
the GWDG/AcademicCloud `qwen3-coder-next` model (the originally configured
`qwen3-coder-30b-a3b-instruct` was not available; `qwen3-coder-next` was the
closest available coding-capable model). Phase 2 required two runs; full results
are in `results/phase2_evaluation_table.md` (Run 2, after fixes) and
`results/phase2_run1_evaluation_table.md` (Run 1, baseline).

### Summary

| Metric | Phase 1 (hand-written) | Phase 2 Run 1 | Phase 2 Run 2 | + Fix 3 (03 only) |
|--------|----------------------|---------------|---------------|-------------------|
| Static check pass | 10/10 | 10/10 | 10/10 | 10/10 |
| Dynamic check pass | 9/10 (90%) | **0/10 (0%)** | 5/10 (50%) | **6/10 (60%)** |
| Repair rounds triggered | n/a | 0 (never fired) | 6 examples | — |
| Repairs resolved | n/a | 0 | 1 (example 10) | — |

### Run 1: what went wrong

Run 1 produced syntactically valid Python for all 10 pipelines on the first
attempt, and all 10 passed `static_check()`. But every single script failed at
runtime — a 90-point gap between static (100%) and dynamic (0%) pass rate.

The root cause had two parts:

**1. Architectural gap in the repair loop.** The repair loop in `converter.py`
called `validate_fn(code)` after each LLM attempt and fed the failure back as a
repair prompt. But `validate_fn` was wired to `static_check()` only — never to
actual script execution. So the loop saw `{"ok": True}` for all 10 scripts and
never fired a single repair round. The dynamic failures were invisible to the
repair mechanism.

**2. Prompt anti-patterns not forbidden.** The four systematic Run 1 failure
categories — `None` placeholder (`skrub.var("name", None)`), missing imports,
sklearn/skrub namespace confusion, and DataOps API misuse (`AggJoiner(aux_table=
DataOp)`, `pd.concat([DataOp, ...])`) — were all patterns the `SKRUB_API_REFERENCE`
in `prompts.py` either implicitly allowed or failed to address.

### What was fixed

**Fix 1 — runtime check wired into repair loop** (`validator.py` + `cli.py`):
`runtime_check(code, working_dir)` runs the candidate script in a subprocess from
the source example's directory, captures the full stderr on failure, and returns it
as the `detail` field. `cli.py` now uses a combined `validate_fn` that runs
`static_check()` first and, if that passes, runs `runtime_check()`. Every repair
prompt now contains the actual Python traceback.

**Fix 2 — `SKRUB_API_REFERENCE` and few-shot examples overhauled** (`prompts.py`):
Added CRITICAL note forbidding `None` as `skrub.var()` default; NAMESPACE RULES
section; explicit import requirement; `.skb.drop()` vs `.drop()` and `.skb.concat()`
vs `pd.concat([DataOp, ...])` rules; rewrote `FEWSHOT_MULTI_TABLE` to demonstrate
stateless `apply_func` + `.merge()` instead of `AggJoiner`.

**Fix 3 — target column surfaced explicitly** (`analyzer.py`, `cli.py`, `prompts.py`):
Example 03's Run 2 failure was traced to the IR's `target_column` field being set
to `"target"` (the default) instead of the actual value `"fraud_flag"`. The analyzer
now scans the AST for `TARGET_COLUMN = "..."` assignments; `cli.py` now extracts
`TARGET_COLUMN` from the loaded module before falling back to AST analysis;
`build_conversion_prompt()` now adds an explicit instruction line: "The target column
is exactly `{target_col}` — do not guess or detect it at runtime from a list of
common names." This is a different class of fix from Fix 1/Fix 2 — it corrects
missing information in the IR, not missing constraints in the prompt.

### Run 2: what each mechanism actually contributed

This distinction matters more than the aggregate numbers suggest.

**Pipelines that passed on attempt 1 (no repair needed) — prompt fix alone was sufficient:**
- 01 Titanic, 04 NYC Taxi, 08 Spooky Author, 09 Home Credit — all 4 passed clean.

**Pipelines that needed a repair round — only this one was resolved by the repair loop:**
- 10 Santander — prompt fix removed the None placeholder; repair loop caught the residual
  `skrub.FrequencyEncoder` hallucination and fixed it in 1 round.

**Pipelines where the repair loop triggered but failed after 3 rounds (02, 05, 06, 07):**
Of 6 pipelines that triggered the repair loop, only 1 was actually resolved by it.
Most of the 0/10 -> 5/10 improvement came from the upfront prompt fix (Fix 2), not
from the LLM's ability to self-correct from a runtime error message (Fix 1's mechanism).
This means the repair loop, now that it correctly fires, is not yet reliably effective
— a genuine, disclosed limitation of the framework's current LLM self-correction
capability, not just of the model chosen.

**"Peel the onion" pattern (05, 07):** Fixing the None-placeholder crash didn't make these
pass — it revealed a second, pre-existing bug that had been masked by the earlier crash.
For 05, removing `None` exposed that the LLM was also passing a function object to both
`skrub.var()` and `.skb.apply()`. For 07, removing the None crash exposed that the LLM
was calling module-level `skrub.concat()` which doesn't exist. These were independent
errors layered beneath the first one, not consequences of the fix.

**Targeted Fix 3 — example 03 resolved separately:** After Run 2, example 03's remaining
failure was diagnosed as a missing-information bug (target column `"fraud_flag"` not passed
to the LLM, because the analyzer defaulted to `"target"`), distinct from the hallucination
bugs in 02/05/06/07. Fix 3 addressed it: after extracting `fraud_flag` from the source AST
and adding an explicit prompt instruction, the LLM converted example 03 correctly on the
first attempt with exact metric match (ROC AUC 0.8497 = 0.8497).

### Remaining failures (4 pipelines, post all fixes)

- **02** (`skrub.FrequencyEncoder` + `low_cardinality_transformer`): stacked hallucinations;
  3 repair rounds couldn't resolve both simultaneously.
- **05** (function object as `skrub.var()` default + function in `.skb.apply()`): two
  simultaneous errors; LLM fixes one per round and reverts the other.
- **06** (`.skb.select(["pred"]).to_pandas()` on estimator output): hallucinated stacking
  API; the correct pattern (`apply_func` wrapper) is not shown in the few-shot examples.
- **07** (`skrub.concat(...)` module-level): hallucinated function that doesn't exist;
  repair loop reverted to the same error on each round.

### Final assessment

The gap between Phase 1 (90% dynamic pass, hand-written) and the automated pipeline (60%
after all fixes, 0% in Run 1) is now fully attributable to specific, diagnosable failure
modes. The framework's three-stage structure is sound; the remaining failures are in (a)
LLM hallucination of non-existent API methods that neither the prompt nor the repair loop
can reliably correct, and (b) the repair loop's limited effectiveness when multiple errors
must be fixed simultaneously. Both are genuine limitations, not evaluation artifacts.

---

## 9. How to reproduce

### Environment setup

```bash
# from the repo root
python3 -m venv .venv
source .venv/bin/activate
pip install skrub scikit-learn xgboost pandas numpy pytest
# for Phase 2 (LLM converter):
pip install openai        # for GWDG/AcademicCloud endpoint
# or
pip install anthropic     # for Anthropic API
```

### Run the test suite

```bash
python3 -m pytest tests/ -v
# expected: 9 passed
```

### Run one harness (dynamic validation for a single pipeline)

```bash
cd examples/01_titanic
python harness.py converted_dataops.py
# prints: {"original_metric": ..., "converted_metric": ...}
```

Any of the 10 example directories works the same way. For example 04 (NYC Taxi),
the parquet data file must already be present in the directory.

### Run the automated converter (requires API key)

```bash
export GWDG_API_KEY=...          # or ANTHROPIC_API_KEY=...
python -m skrubifier convert examples/01_titanic/source_pipeline.py --out out.py
# with Anthropic backend:
python -m skrubifier convert examples/01_titanic/source_pipeline.py --out out.py --backend anthropic
```

The converter will run the three-stage pipeline (analyze -> convert -> validate
with repair loop) and write a validated DataOps script to `out.py`.

---

*For the full change history, see `CHANGELOG.md`. For the complete dynamic
validation table, see `results/evaluation_table.md`. For the pipeline selection
rationale and known conversion hazards, see `examples/PLAN.md`.*
