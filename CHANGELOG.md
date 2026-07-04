# Skrubifier — Change Log

Entries added as work progresses (not retroactively). Each non-trivial change
records what, why, and how to verify. See `WRITEUP.md` for the project overview.

---

## Phase 1

### [examples/01–05 — data] Synthetic datasets created for examples missing CSVs

**What changed:** Created synthetic `train.csv` for examples 01 (Titanic),
02 (House Prices), 05 (Otto Group) and `baskets.csv`/`products.csv` for
example 03 (Credit Fraud). Real Kaggle data is not downloadable in this
environment; synthetic data matches exact column names, dtypes, and
distributional shape of the originals so the harness runs end-to-end.

**Why:** Dynamic-checking all six examples is required to establish verified references; only
example 04 (NYC Taxi) ships with real parquet data. The synthetic files are
small (500–2000 rows) but large enough that the metrics are non-trivial.

**How to verify:**
```
ls examples/01_titanic/train.csv
ls examples/02_house_prices_xgb/train.csv
ls examples/03_credit_fraud_multitable/baskets.csv
ls examples/05_otto_group/train.csv
```

---

### [examples/02–05 — harness.py] Added missing harness files

**What changed:** Created `harness.py` for examples 02, 03, 04, and 05.
Only example 01 shipped with a harness; the others were hand-written
`converted_dataops.py` files with no corresponding evaluation harness.

**Why:** Dynamic validation requires running `python harness.py converted_dataops.py`
for all six examples. Without harnesses, dynamic validation is impossible.

**How to verify:**
```
cd examples/02_house_prices_xgb && python harness.py converted_dataops.py
cd examples/03_credit_fraud_multitable && python harness.py converted_dataops.py
cd examples/04_nyc_taxi_fare && python harness.py converted_dataops.py
cd examples/05_otto_group && python harness.py converted_dataops.py
```

---

### [examples/02_house_prices_xgb/converted_dataops.py] Fix: TableVectorizer parameter rename

**What changed:** Changed `skrub.TableVectorizer(high_cardinality_transformer=TargetEncoder())`
to `skrub.TableVectorizer(high_cardinality=TargetEncoder())`.

**Why:** The harness failed with `TypeError: TableVectorizer.__init__() got an
unexpected keyword argument 'high_cardinality_transformer'`. In skrub 0.9.0,
the parameter was renamed from `high_cardinality_transformer` to
`high_cardinality`. The hand-written script used the old name.

**How to verify:**
```
cd examples/02_house_prices_xgb && python harness.py converted_dataops.py
```
Should print JSON with both metrics.

---

### [skrubifier/analyzer.py] Fix: Update TargetEncoder SKRUB_HINTS for skrub 0.9

**What changed:** Updated `SKRUB_HINTS["TargetEncoder"]` from the old
`low_cardinality_transformer=skrub.TargetEncoder()` form (wrong slot, wrong
parameter name) to a hint referencing `high_cardinality=<encoder>` and noting
the skrub 0.9+ parameter name.

**Why:** The old hint would have caused the LLM to generate code with an
incorrect TableVectorizer parameter name.

**How to verify:**
```
python3 -m pytest tests/test_analyzer.py -v
```

---

### [examples/02_house_prices_xgb/converted_dataops.py] Fix: apply_func after estimator node

**What changed:** Replaced the `pred_log.skb.apply_func(np.expm1)` + `learner =
pred.skb.make_learner()` pattern with
`TransformedTargetRegressor(regressor=XGBRegressor(...), func=np.log1p,
inverse_func=np.expm1)` wrapped in a single `.skb.apply(model, y=y)` call.
The `y` mark was moved to the original SalePrice (not log1p(SalePrice)).

**Why:** In skrub 0.9.0, during `learner.fit()`, an estimator DataOp node
evaluates to the FITTED ESTIMATOR OBJECT (not predictions). Chaining
`apply_func(np.expm1)` after it therefore calls `np.expm1(XGBRegressor)`,
which fails: "loop of ufunc does not support argument 0 of type XGBRegressor."
The correct pattern for target-space transforms on final estimator outputs is
`TransformedTargetRegressor`, which handles the log1p/expm1 internally.

**How to verify:**
```
cd examples/02_house_prices_xgb && python harness.py converted_dataops.py
```
Should print: `{"original_metric": ~0.772, "converted_metric": ~0.777}`

---

### [examples/03_credit_fraud_multitable/converted_dataops.py] Fix: AggJoiner predict-time behavior + basket_ID leakage

**What changed:** Replaced both `skrub.AggJoiner(...)` calls with a stateless
`_agg_products(products_df)` function applied via `products.skb.apply_func(_agg_products)`,
followed by `baskets.merge(agg, on="basket_ID", how="left")` using the DataOp
`.merge()` operator. Also changed `X = baskets_with_counts.drop(columns=["fraud_flag"])`
to explicitly selecting only the 4 derived feature columns.

**Why (AggJoiner):** `skrub.AggJoiner` is a sklearn-style transformer: it FITS
the aggregation on training aux_table and CACHES it for `transform()` calls.
Test baskets not in training products get NaN -> model predicts constant -> AUC=0.5.
Discovered by inspecting probas: `[9.99204e-01, 7.96e-04]` for every test basket.
The source pipeline re-computes aggregations fresh at prediction time (`build_features`
is called again), so the DataOps equivalent must also be stateless.

**Why (basket_ID leakage):** After fixing the AggJoiner issue, the model still
produced AUC=0.5. Root cause: `drop(columns=["fraud_flag"])` left basket_ID,
customer_age, and the derived features in X. basket_ID is an ID column:
training baskets have IDs 1-480, test baskets 481-600. Model memorises training
IDs and produces constant predictions for unseen test IDs.

**How to verify:**
```
cd examples/03_credit_fraud_multitable && python harness.py converted_dataops.py
```
Should print: `{"original_metric": ~0.850, "converted_metric": ~0.850}`

---

### [examples/05_otto_group/source_pipeline.py, converted_dataops.py, harness.py] Fix: XGBoost 3.x string label rejection

**What changed:**
- Added `CLASSES = ["Class_1"..."Class_9"]` and `LabelEncoder` in
  `source_pipeline.py.__main__` to encode before fitting.
- Added `apply_func(lambda s: s.str.replace('Class_', '').astype(int) - 1)` on
  the target column in `converted_dataops.py` to encode 'Class_N' -> N-1 as an
  integer within the DataOps DAG.
- Updated `harness.py` to use `LabelEncoder` before fitting the sklearn pipeline
  and to convert the binary label matrix with `label_binarize`.

**Why:** XGBoost 3.x with `objective="multi:softprob"` + `num_class=9` requires
integer class labels (0..8). The real Kaggle dataset uses string labels
('Class_1'..'Class_9'). Fitting raised: "Invalid classes inferred from unique
values of `y`. Expected: [0 1 2 ... 8], got ['Class_1' 'Class_2' ...]".

**How to verify:**
```
cd examples/05_otto_group && python harness.py converted_dataops.py
```
Should print: `{"original_metric": ~0.815, "converted_metric": ~0.820}`

---

### [examples/06_allstate_claims_severity — new] Created example 06

**What changed:** Created `source_pipeline.py`, `converted_dataops.py`,
`harness.py`, and synthetic `train.csv` (2000 rows, 116 cat + 14 num features)
for the Allstate Claims Severity stacking pattern.

**Why:** The deliverable requires 6 examples; only 5 existed. Example 06 covers
the StackingRegressor conversion hazard (PLAN.md #8).

**Bugs found during creation:**
- `skrub.concat()` does not exist; the module-level function is missing.
  Correct method is `.skb.concat([...], axis=1)` on a DataOp instance.
- `.skb.concat()` requires DataFrame inputs, not Series. Each base-model
  prediction (a Series) must be wrapped with
  `.skb.apply_func(lambda s: pd.DataFrame({"pred_xgb": s}))` first.

**Known limitation:** Dynamic check FAILS (delta 0.037, tol 0.020). Root cause:
sklearn StackingRegressor uses OOF predictions (no leakage); the DataOps version
uses in-fold base predictions for the meta-learner (mild leakage). No DataOps
OOF stacking primitive exists in skrub 0.9. Documented as a framework limitation.

**How to verify:**
```
cd examples/06_allstate_claims_severity && python harness.py converted_dataops.py
```
Expected: `{"original_metric": ~0.644, "converted_metric": ~0.608}` (delta > tol, documented failure).

---

### [examples/07-10 — new] Created examples 07–10 (Phase 1.3)

**What changed:** Created four new pipeline examples from PLAN.md:

- `07_random_acts_of_pizza`: text+tabular binary classif. (TF-IDF -> TableVectorizer/MinHashEncoder)
- `08_spooky_author`: pure-text 3-class classif. (TF-IDF -> MinHashEncoder)
- `09_home_credit`: multi-table credit scoring (same apply_func+merge pattern as 03)
- `10_santander`: pure-numeric SelectKBest (sklearn Pipeline wrapped in `.skb.apply()`)

Each example includes: `source_pipeline.py`, `converted_dataops.py`, `harness.py`,
and synthetic data matching the real dataset's column names/dtypes.

**Notes:**
- Example 07: `skrub.TextEncoder` requires `sentence_transformers` (not installed).
  Converted uses `TableVectorizer` which auto-routes string cols to `MinHashEncoder`.
  `TextEncoder` is documented in SKRUB_API_REFERENCE but marked as requiring optional deps.
- Example 09: Initial synthetic data had near-zero signal (AUC ~0.5 for both
  models). Regenerated with bureau-based features strongly correlated with default.
- Example 10: Converted wraps the full sklearn Pipeline in a single `.skb.apply()`;
  metrics are bit-identical since no DataOps transformation changes the model.

**How to verify:**
```
cd examples/07_random_acts_of_pizza && python harness.py converted_dataops.py
cd examples/08_spooky_author && python harness.py converted_dataops.py
cd examples/09_home_credit && python harness.py converted_dataops.py
cd examples/10_santander && python harness.py converted_dataops.py
```

---

### [tests/test_analyzer.py] Updated static check list to cover all 10 examples

**What changed:** Added examples 06–10 to the
`test_static_check_accepts_all_hand_written_converted_scripts` test.

**Why:** The test previously only covered examples 01–05. All converted scripts
must pass the static validator.

**How to verify:**
```
python3 -m pytest tests/test_analyzer.py::test_static_check_accepts_all_hand_written_converted_scripts -v
```

---

### [results/evaluation_table.md — new] Phase 1 evaluation table created

**What changed:** Created `results/evaluation_table.md` with one row per pipeline
(01–10): name, domain, dataset size, ML task, static check, dynamic check
(original/converted metrics + delta + tolerance), pass/fail, and notes.

**Why:** This file is required grading evidence.

**How to verify:** Review `results/evaluation_table.md` and cross-reference with
`CHANGELOG.md` entries above.

---

### [examples/08_spooky_author] Regenerated synthetic data + fixed converted_dataops.py to remove ceiling-effect AUC=1.0

**What changed:**

1. `train.csv` regenerated. Old data had fully disjoint per-class vocabularies
   (EAP: `horror/soul/dark`; HPL: `eldritch/cyclopean/void`; MWS: `monster/beauty/love`)
   — trivially separable by any word-feature model, giving AUC=1.0 for both source
   and converted. New data mixes four word pools per sample: common function words
   (~30%), shared content words (~15%), author-preferring words (~32%), and cross-class
   noise words (~23%), producing realistic class overlap. New metrics: AUC 0.963 vs 0.963.

2. `converted_dataops.py` updated. Old version used `skrub.MinHashEncoder(n_components=100)`,
   which operates on character n-grams and produced AUC 0.856 on the new data — a delta
   of 0.107 vs the source's 0.963, well outside tolerance (0.029). MinHashEncoder is
   inappropriate for word-vocabulary text classification; TF-IDF's IDF weighting on
   individual words gives it a structural advantage that can't be closed by increasing
   n_components. Fix: route the text through the DAG as a Series via
   `.skb.apply_func(lambda df: df["text"])` and then apply the same
   `TfidfVectorizer(max_features=5000, ngram_range=(1,2), ...)` as the source via
   `.skb.apply()`. The TfidfVectorizer fits on training text and transforms at predict
   time — correct behaviour within the DataOps DAG, no test-set leakage.
   New delta: 0.000 (identical algorithm).

**Alternative tried (documented skrub limitation):** Before switching to TfidfVectorizer,
`skrub.MinHashEncoder(n_components=100)` (and `TableVectorizer`'s default high-cardinality
routing, which also routes strings through MinHashEncoder) were tried as the "pure skrub"
conversion. On the regenerated overlapping-vocabulary data: original=0.963, converted=0.856,
delta=0.107 vs tolerance=0.029 — a hard FAIL. This is structural, not a tuning issue:
MinHashEncoder hashes character n-grams of the full text string into a fixed-size dense
vector; it cannot replicate TF-IDF's per-token IDF weighting, which is what discriminates
between authors on bag-of-words text. The gap does not close with more components.
`TextEncoder` (sentence-transformer embeddings) would likely close it but requires an
optional dependency not present in this environment. This result is preserved in
`converted_dataops.py`'s docstring as a disclosed limitation of the skrub-native text
routing path for this pipeline class.

**Why:** The original ceiling-effect result (AUC=1.0 both sides) was weak evidence
for conversion correctness — a random prediction that happened to sort correctly would
also pass. The new data gives meaningful non-trivial metrics. Using TfidfVectorizer
in the DataOps script is the same pattern as example 10 (wrapping a full sklearn
estimator in `.skb.apply()`): it demonstrates that DataOps correctly handles stateful
sklearn transformers, including fit-then-transform semantics.

**How to verify:**
```
cd examples/08_spooky_author && python harness.py converted_dataops.py
```
Expected: `{"original_metric": ~0.963, "converted_metric": ~0.963}`

---

### [tests/test_analyzer.py, skrubifier/analyzer.py] Add stacking ensemble expansion + regression test

**What changed:** Added `test_estimator_analyzer_captures_stacking_base_estimators` to
`tests/test_analyzer.py` and the corresponding `_expand_ensemble_into_steps` helper
(plus four new SKRUB_HINTS entries) to `skrubifier/analyzer.py`.

**Why:** `_params_of` deliberately drops non-JSON-primitive values; this silently
swallowed `StackingRegressor.estimators` (a list of tuples) and `StackingRegressor.final_estimator`
(an estimator object), leaving the IR with no record of the sub-models. The fix
detects `StackingRegressor`, `StackingClassifier`, `VotingRegressor`, and
`VotingClassifier` at the final-estimator position and emits one `StepIR` per base
estimator (`applies_to="{Cls}.base_estimator"`) plus one for `final_estimator`
(`applies_to="{Cls}.final_estimator"`), and appends a note that no single skrub
primitive exists. The test was written when example 06 was developed in a separate
working copy but was omitted from the files handed into this repo.

**How to verify:**
```
python3 -m pytest tests/test_analyzer.py::test_estimator_analyzer_captures_stacking_base_estimators -v
```

---

## Phase 2

### Setup

- **Model used:** `qwen3-coder-next` (GWDG/AcademicCloud SAIA endpoint,
  `https://chat-ai.academiccloud.de/v1`)
  — The originally configured default `qwen3-coder-30b-a3b-instruct` was not
  present in the `/v1/models` response; `qwen3-coder-next` was selected as the
  closest available coding-capable model. Updated in `skrubifier/converter.py`
  and `skrubifier/cli.py`.
- **Pipelines run:** All 10 (examples 01–10)
- **LLM output location:** `results/llm_conversions/`
- **Full results table:** `results/phase2_evaluation_table.md`

### Results

| Metric | Value |
|--------|-------|
| Static check pass | **10/10** |
| Dynamic check pass | **0/10** |
| Average repair rounds | **1.0** (no repairs — static always passed on attempt 1) |

All 10 LLM-generated scripts passed `static_check()` on the first attempt and
required no repair rounds. However, all 10 fail when executed. The static
validator's pass signal was insufficient to indicate runtime correctness — it
only checks Python syntax and `.skb`-namespace presence, not semantic
correctness of DataOps API usage.

### Systematic failure categories

1. **`None` placeholder anti-pattern** (4 failures — 01, 07, 10, 05):
   `skrub.var("name", None)` followed by pandas `.drop()` / `.skb.select()`.
   Skrub runs preview evaluation at script load using the default value;
   `None.drop(...)` immediately raises `AttributeError`.

2. **Missing import** (1 failure — 02):
   `pd.read_csv(...)` used in `skrub.var()` default without `import pandas as pd`.

3. **Sklearn/skrub namespace confusion** (2 failures — 04, 08):
   `skrub.StandardScaler()` (doesn't exist) and `skrub.TextEncoder(encoding=...,
   max_features=...)` (TextEncoder doesn't accept TF-IDF parameters).

4. **DataOps API misuse** (3 failures — 03, 06, 09):
   `AggJoiner(aux_table=DataOp)`, `pd.concat([DataOp, ...])` instead of `.skb.concat()`,
   and `skrub.var("app", app_df)` with `app_df` undefined.

### Key finding (Run 1)

The 90-point gap between static pass (100%) and dynamic pass (0%) identifies that
`static_check()` is insufficient as the sole quality gate. The repair loop never
fired because it relies on static check failures as the trigger — all semantic
errors passed through undetected. See `results/phase2_run1_evaluation_table.md` for
per-pipeline detail.

---

### [skrubifier/validator.py] Fix 1 — `runtime_check()` added

**What changed:** Added `runtime_check(code, working_dir, timeout_s=60)` function.
Writes the candidate script to a temp file in the example's directory, runs it in
a subprocess with `cwd=working_dir`, captures full stderr on non-zero exit.
Returns `{"ok": False, "detail": "<traceback>"}` on failure or
`{"ok": True, "detail": "static and runtime checks passed"}` on success.

**Why:** Run 1 showed 0/10 dynamic pass despite 10/10 static pass because the
repair loop was never fed real execution errors. All four Run 1 failure categories
(None placeholder, missing imports, namespace confusion, API misuse) are detectable
by actually running the candidate script.

**How to verify:**
```
python -c "
from skrubifier.validator import runtime_check
import os
good = 'import skrub\ndf = __import__(\"pandas\").DataFrame({\"a\":[1]})\nv = skrub.var(\"x\", df)\nl = v.skb.mark_as_X().skb.apply(__import__(\"sklearn.linear_model\", fromlist=[\"Ridge\"]).Ridge()).skb.make_learner()\n'
print(runtime_check(good, os.getcwd()))
"
```

---

### [skrubifier/cli.py] Fix 1 — combined `validate_fn` wires runtime check into repair loop

**What changed:** The `validate_fn` lambda passed to `converter.convert()` now runs
`static_check(code)` first and, if that passes, runs `runtime_check(code, source_dir)`
where `source_dir = os.path.dirname(os.path.abspath(args.source))`. The real traceback
becomes the `failure` field in the `REPAIR_TEMPLATE`, so the LLM sees the actual error
on each repair round.

**Why:** Without this, the repair loop had a Boolean blind spot: the only repair signal
was `static_check()`, which always returned `ok=True` for semantically wrong but
syntactically valid scripts.

---

### [skrubifier/prompts.py] Fix 2 — `SKRUB_API_REFERENCE` and few-shot examples overhauled

**What changed:**
- Added CRITICAL note forbidding `skrub.var("name", None)` — second argument evaluated
  eagerly at script definition time; must be a real loaded DataFrame.
- Added NAMESPACE RULES: sklearn classes (`StandardScaler`, `SelectKBest`, etc.) are not
  in `skrub`; `TextEncoder` takes no TF-IDF params; `high_cardinality` not
  `high_cardinality_transformer`.
- Added explicit import rule: every used module must be imported.
- Clarified `.skb.drop()` vs `.drop()`, `.skb.concat()` vs `pd.concat([DataOp, ...])`.
- Clarified `AggJoiner.aux_table` must be a real DataFrame, not a DataOp.
- Rewrote `FEWSHOT_MULTI_TABLE` to demonstrate stateless `apply_func` + `.merge()`
  pattern with explanation of why it's preferred over `AggJoiner`.
- Rewrote `FEWSHOT_SINGLE_TABLE` to load a real DataFrame before `skrub.var()`.

**Why:** Run 1's four failure categories were all predictable from the prompt's gaps.
This overhaul makes the correct patterns explicit and the common anti-patterns forbidden.

---

### Phase 2 Run 2 — results

Ran all 10 pipelines again after Fix 1 + Fix 2. Full results in
`results/phase2_evaluation_table.md`.

| Metric | Run 1 | Run 2 | After Fix 3 |
|--------|-------|-------|-------------|
| Static check pass | 10/10 | 10/10 | 10/10 |
| Runtime check pass | 0/10 | 5/10 | **6/10** |
| Dynamic harness pass | 0/10 | 5/10 | **6/10** |
| Repair rounds triggered | 0 | 6 examples | — |
| Repairs resolved | 0 | 1 (example 10) | — |

**Mechanism breakdown:** 4 of the 5 Run 2 passes came from the prompt fix alone (Fix 2 —
attempts=1, no repair needed): 01, 04, 08, 09. Only 1 pass (example 10) came from the
repair loop (Fix 1) catching a residual FrequencyEncoder hallucination. The repair loop
fired for 5 other examples (02, 03, 05, 06, 07) but resolved none within 3 rounds.

**Still failing after Run 2 (02, 05, 06, 07):** Hallucinated skrub classes/methods
(`FrequencyEncoder`, `skrub.concat`, `.skb.select().to_pandas()`) or stacked
function-vs-estimator errors. The repair loop reached its 3-round limit without resolving any.

---

### [skrubifier/analyzer.py, cli.py, prompts.py] Fix 3 — target column extracted from AST and surfaced explicitly

**What changed:**
- `analyzer.py`: Added `_ast_target_column()` helper that scans the AST for
  `TARGET_COLUMN = "..."` / `TARGET_COL = "..."` assignments. `analyze_source()` now
  calls this when no explicit `target_column` is passed (changed signature from
  `target_column: str = "target"` to `target_column: Optional[str] = None`).
- `cli.py`: When falling back to AST analysis, now extracts `TARGET_COLUMN` from the
  already-loaded module (`getattr(mod, "TARGET_COLUMN", None)`) before calling
  `analyze_source()`. This means the module's own definition takes precedence over
  the AST scan.
- `prompts.py`: `build_conversion_prompt()` now parses the IR JSON to extract
  `target_column` and adds an explicit instruction line before the few-shot examples:
  `"The target column is exactly "{target_col}" — do not guess or detect it at runtime."`

**Why:** Example 03's Run 2 failure was not a hallucination bug. The root cause was that
`analyze_source()` defaulted `target_column="target"`, so the IR said `"target_column":
"target"` even though the source declared `TARGET_COLUMN = "fraud_flag"`. The LLM saw the
correct value in the raw source (`original_source` in the prompt) but generated a generic
detection loop instead of using it. Adding the explicit instruction and fixing the IR breaks
that pattern.

**How to verify:**
```
# Confirm analyzer extracts the right value for example 03
cd examples/03_credit_fraud_multitable
python -c "
import sys; sys.path.insert(0, '../..')
from skrubifier.analyzer import analyze_source
print(analyze_source('source_pipeline.py').target_column)
# should print: fraud_flag
"
```

### Phase 2 Fix 3 result — example 03

Re-ran example 03 only (no full re-run):
- `target_column` in IR now: `fraud_flag` (was `target`)
- Static+runtime check: Pass — attempt 1, no repair needed
- Dynamic harness: Pass — ROC AUC 0.8497 = 0.8497 (exact match)
- This brings the final dynamic pass rate to **6/10**.
