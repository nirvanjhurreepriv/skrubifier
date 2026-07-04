# Skrubifier — Evaluation Table (Phase 1: Hand-Written References)

## Data sourcing vs. pipeline sourcing

The assignment requires source *pipelines* to come from MLE-Bench and
Kaggle — it does not require runtime *data* to be the original competition
data. All 10 pipelines in this project are adapted from real, published
MLE-Bench/Kaggle solutions (see each `source_pipeline.py`'s docstring for
its specific source). Runtime data is real where available (example 04,
NYC Taxi Fare, ships with actual competition data) and synthetic — matched
to the original dataset's column names, dtypes, and approximate
distributional shape — where the real dataset wasn't downloadable in this
environment (examples 01–03, 05–10; see `results/evaluation_table.md` for
the per-pipeline breakdown). This is disclosed per-pipeline rather than
uniformly, since synthetic data affects the strength of the correctness
evidence differently across pipelines — e.g. example 08's synthetic text
being perfectly separable produced a ceiling-effect result (AUC=1.0 both
sides) that demonstrates less than a comparable real-data result would.

---

All 10 pipelines were dynamic-validated by running each example's `harness.py`
against `converted_dataops.py`. Tolerance = `max(0.02, 0.03 * |original_metric|)`.

## Results

| # | Name | Domain | Dataset | ML Task | Metric | Static OK | Original | Converted | Delta | Tol | Dynamic OK | Notes |
|---|------|--------|---------|---------|--------|-----------|----------|-----------|-------|-----|------------|-------|
| 01 | Titanic | Survival | Synthetic (~891 rows) | Binary classif. | ROC AUC | Pass | 0.768 | 0.782 | 0.014 | 0.023 | Pass | Synthetic data; TableVectorizer subsumes ColumnTransformer |
| 02 | House Prices XGBoost | Real estate regression | Synthetic (~1460 rows) | Regression | R² (log scale) | Pass | 0.772 | 0.777 | 0.005 | 0.023 | Pass | **Bug fixed**: `high_cardinality_transformer` -> `high_cardinality` (skrub 0.9 rename); **Bug fixed**: `apply_func(np.expm1)` after estimator node fails -> switched to `TransformedTargetRegressor` |
| 03 | Credit Fraud Multitable | Fraud detection | Synthetic (600 baskets, 2400 products) | Binary classif. | ROC AUC | Pass | 0.850 | 0.850 | 0.000 | 0.026 | Pass | **Bug fixed**: AggJoiner uses cached training aggregations at predict time (test baskets get NaN -> AUC=0.5); fixed by using `.skb.apply_func()` stateless aggregation + DataOp `.merge()` instead of AggJoiner; **Bug fixed**: original `.drop(columns=["fraud_flag"])` kept basket_ID as feature (memorises training IDs -> AUC=0.5 after fix 1) |
| 04 | NYC Taxi Fare | Geo/transport regression | **Real data** (500k train, subsampled to 10k) | Regression | R² | Pass | 0.656 | 0.656 | 0.000 | 0.020 | Pass | Real data; `eval_mode()+if_else()` train-only row filter works correctly |
| 05 | Otto Group | Multi-class product classif. | Synthetic (1000 rows, 93 features) | 9-class classif. | ROC AUC (macro OvR) | Pass | 0.815 | 0.820 | 0.005 | 0.024 | Pass | **Bug fixed**: XGBoost 3.x requires integer labels; added `apply_func(lambda s: s.str.replace('Class_', '').astype(int) - 1)` on target; converted uses choose_from grid search vs source's single config |
| 06 | Allstate Claims Severity | Insurance regression | Synthetic (2000 rows, 130 features) | Regression | R² (log scale) | Pass | 0.644 | 0.608 | 0.037 | 0.020 | Fail | **Known limitation**: StackingRegressor (sklearn) uses OOF CV predictions for meta-learner (no leakage); DataOps version uses in-fold predictions (mild leakage), so converted metric is slightly lower. No DataOps-native OOF stacking primitive available in skrub 0.9. `skrub.concat()` not available; uses `.skb.concat()` instead. |
| 07 | Random Acts of Pizza | Text+tabular classif. | Synthetic (5000 rows) | Binary classif. | ROC AUC | Pass | 0.638 | 0.638 | 0.000 | 0.019 | Pass | **Note**: `TextEncoder` requires `sentence_transformers` (not installed); converted uses `TableVectorizer` which routes string cols to `MinHashEncoder`. Metrics match well. |
| 08 | Spooky Author | NLP 3-class classif. | Synthetic (5000 rows) | 3-class classif. | ROC AUC (macro OvR) | Pass | 0.963 | 0.963 | 0.000 | 0.029 | Pass | Data regenerated with shared vocabulary across classes (common + class-preferring + cross-class noise words) to remove ceiling effect. Converted uses TfidfVectorizer via `.skb.apply_func(lambda df: df["text"])` -> `.skb.apply(TfidfVectorizer(...))` — same algorithm as source, routed through DataOps DAG so test-set vocabulary is never leaked. `multi_class='multinomial'` deprecated in sklearn 1.5 (FutureWarning; not a breaking change). |
| 09 | Home Credit Default Risk | Credit scoring | Synthetic (3000 apps, 12000 bureau records) | Binary classif. | ROC AUC | Pass | 0.881 | 0.883 | 0.002 | 0.026 | Pass | Same apply_func+merge pattern as example 03; AggJoiner not used. Regenerated data (initial synthetic had near-zero signal -> AUC ~0.5). |
| 10 | Santander Customer Transaction | Banking binary classif. | Synthetic (5000 rows, 200 features) | Binary classif. | ROC AUC | Pass | 0.573 | 0.573 | 0.000 | 0.020 | Pass | Converted wraps full sklearn Pipeline (StandardScaler+SelectKBest+LR) in single `.skb.apply()`; identical metrics confirm pass-through fidelity. |

---

## Summary

| | Count |
|---|---|
| Pipelines built | 10/10 |
| Static check pass | 10/10 |
| Dynamic check pass (within tolerance) | 9/10 |
| Dynamic check fail (documented limitation) | 1/10 (example 06) |

**Failure detail (example 06):** Allstate stacking — delta 0.037 vs tolerance 0.020. Root cause: sklearn StackingRegressor uses OOF (cross-validated) predictions for the meta-learner, avoiding leakage. The DataOps version applies all branches on the same training set, giving the meta-learner in-fold predictions (mild leakage). No DataOps-native OOF stacking primitive exists in skrub 0.9. This is a genuine framework limitation for this pipeline class — documented, not hidden.

---

## API Bugs Found During Dynamic Validation

| Bug | Affected Example | Fix Applied |
|-----|-----------------|-------------|
| `TableVectorizer` parameter renamed: `high_cardinality_transformer` -> `high_cardinality` in skrub 0.9 | 02 | Updated converted_dataops.py + analyzer.py hint |
| `apply_func(np.expm1)` after estimator `apply()` node fails during fit (node holds fitted estimator, not predictions) | 02 | Switched to `TransformedTargetRegressor(func=log1p, inverse_func=expm1)` |
| `AggJoiner` uses cached training aggregations at predict time — test entities not in training aux table get NaN | 03, 09 | Replaced AggJoiner with `apply_func(_agg_func)` + DataOp `.merge()` pattern |
| `AggJoiner.drop(columns=["fraud_flag"])` left basket_ID in features — model memorises training basket IDs | 03 | Explicitly selected only the 4 derived feature columns |
| XGBoost 3.x: string class labels ('Class_1'..'Class_9') rejected with `multi:softprob` objective | 05 | Added `apply_func` label encoding on target + `LabelEncoder` in harness |
| `skrub.concat()` does not exist — `.skb.concat()` is the correct method | 06 | Used `pred.skb.concat([...], axis=1)` |
| `TextEncoder` requires `sentence_transformers` (optional dep not installed) | 07 | Used `TableVectorizer` (routes strings to `MinHashEncoder` automatically) |

---

## Phase 2 Status

**Blocked**: Neither `GWDG_API_KEY` nor `ANTHROPIC_API_KEY` is set in the
environment. Phase 2 (automated LLM conversion) cannot proceed. The evaluation
table above covers Phase 1 (hand-written reference pipelines) only.
