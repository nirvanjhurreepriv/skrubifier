# Skrubifier — Phase 2 Evaluation Table

## Quick Summary

| | Run 1 | Run 2 | After Fix 3 (03 only) |
|---|-------|-------|----------------------|
| Static+runtime pass | 0/10 | 5/10 | 6/10 |
| Dynamic harness pass | 0/10 | 5/10 | **6/10** |

---

## Which Fix Resolved Which Failure Category

_(Corrected attribution — derived from Run 1 error messages, not assumption.)_

| Run 1 failure category | Affected examples | Fix applied | Run 2 outcome |
|------------------------|------------------|-------------|---------------|
| `None` placeholder (`skrub.var("name", None)`) | 01, 05, 07, 10 | Fix 2: prompts.py — forbid None; always load real DataFrame | 01 Pass, attempt 1; 10 Pass, attempt 2 (repair loop caught FrequencyEncoder); 05 Fail, deeper error revealed; 07 Fail, deeper error revealed |
| Missing `import pandas as pd` | 02 | Fix 2: prompts.py — Rule 2, all imports required | import fixed, but `skrub.FrequencyEncoder` hallucination emerged -> Fail after 3 repairs |
| Sklearn/skrub namespace confusion | 04, 08 | Fix 2: prompts.py — NAMESPACE RULES section | 04 Pass, attempt 1; 08 Pass, attempt 1 |
| DataOps API misuse (AggJoiner anti-pattern, `pd.concat`, forward ref) | 03, 06, 09 | Fix 2: prompts.py — AggJoiner clarification, `.skb.concat()` rule, stateless few-shot | 09 Pass, attempt 1 (forward ref fixed); 06 Fail, different error; 03 Fail in Run 2 -> fixed by Fix 3 |
| Target column buried in JSON blob / not extracted by analyzer | 03 | Fix 3: `_ast_target_column()` in `analyzer.py`; `TARGET_COLUMN` extraction in `cli.py`; explicit instruction in `build_conversion_prompt()` | 03 Pass, attempt 1 (metric 0.8497 = 0.8497) |
| Repair loop not wired to runtime errors | (all 10) | Fix 1: `runtime_check()` in `validator.py` + combined `validate_fn` in `cli.py` | 10 resolved in 1 repair; 02, 05, 06, 07 hit 3-round limit without resolution |

**Correction note on Run 1 failure list:** Example 09's Run 1 failure was a circular variable
reference (`app_df = skrub.var("app", app_df)` — using `app_df` before it was defined),
categorized under "DataOps API misuse." Example 10's Run 1 failure was the None placeholder.
Neither should appear in the other's category, as they were in the original table.

---

## What Mechanism Actually Produced Each Pass

This is the key breakdown the summary numbers obscure.

| Pipeline | Run 2 pass mechanism | Attempts |
|----------|----------------------|----------|
| 01 Titanic | **Prompt fix alone** — None placeholder removed, no repair needed | 1 |
| 04 NYC Taxi | **Prompt fix alone** — namespace confusion removed, no repair needed | 1 |
| 08 Spooky Author | **Prompt fix alone** — namespace confusion removed, no repair needed | 1 |
| 09 Home Credit | **Prompt fix alone** — forward reference fixed, no repair needed | 1 |
| 10 Santander | **Repair loop** — prompt fix removed None placeholder, repair loop fixed residual FrequencyEncoder hallucination | 2 |
| 03 Credit Fraud | **Framework fix (Fix 3)** — target column extracted from AST + explicit prompt instruction | 1 |

**The core finding:** of 6 pipelines that now pass, **4 passed on attempt 1 from the prompt fix
alone (Fix 2)**, **1 required the repair loop (Fix 1)**, and **1 required a targeted framework
fix (Fix 3)**. The repair loop mechanism (Fix 1) contributed exactly one resolution. For the
4 remaining failures (02, 05, 06, 07), the repair loop triggered but failed to fix any of them
within 3 rounds.

**"Peel the onion" pattern (05, 07):** Fixing the None placeholder crash didn't make these two
pass — it revealed a second, pre-existing bug that had been masked by the earlier crash. For
example 05, removing `skrub.var("data", None)` exposed that the LLM was also passing a plain
function to `.skb.apply()`. For example 07, removing the None-based `.drop()` crash exposed
that the LLM was calling module-level `skrub.concat()` which doesn't exist. These are
independent errors that layered beneath the first one, not consequences of the fix.

---

## Setup

- **Model**: `qwen3-coder-next` (GWDG/AcademicCloud SAIA endpoint)
- **Backend**: `--backend gwdg`; **Max repair rounds**: 3
- **Static check**: AST parse + `.skb`-namespace presence + required call checks
- **Runtime check**: subprocess execution from source example's directory; full stderr as repair signal
- **Metric tolerance**: `max(0.02, 0.03 * |original_metric|)`
- **Harness**: `examples/NN_name/harness.py`; for scripts using generic var names, adapted runners in `results/run2_harness_NN.py`

---

## Per-Pipeline Results (Final — after all three fixes)

| # | Name | Static+RT OK | Attempts | Dynamic OK | Original | Converted | Error / Root Cause |
|---|------|-------------|----------|------------|----------|-----------|-------------------|
| 01 | Titanic | Pass | 1 | Pass | 0.7684 | 0.7825 | Prompt fix alone. |
| 02 | House Prices XGBoost | Fail | 4 | Fail | — | — | `skrub.FrequencyEncoder` doesn't exist; also `low_cardinality_transformer` (old param name). Both errors stacked; 3 repairs could not resolve. |
| 03 | Credit Fraud Multitable | Pass | 1 | Pass | 0.8497 | 0.8497 | **Fixed by Fix 3** (target column `fraud_flag` now extracted from AST and surfaced explicitly). |
| 04 | NYC Taxi Fare | Pass | 1 | Pass | 0.6561 | 0.6561 | Prompt fix alone. |
| 05 | Otto Group | Fail | 4 | Fail | — | — | `skrub.var("data", load_data)` passes a function object; `.skb.apply(preprocess)` passes a plain function — should be `.skb.apply_func()`. Two errors masked by the Run 1 None crash; both revealed in Run 2 but not jointly fixed by 3 repair rounds. |
| 06 | Allstate Claims Severity | Fail | 4 | Fail | — | — | `xgb_pred.skb.select(["pred"]).to_pandas()` — neither `.skb.select(list)` on a Series nor `.to_pandas()` on estimator output exist. Stacking pattern is hallucinated; LLM cannot recover within 3 rounds. |
| 07 | Random Acts of Pizza | Fail | 4 | Fail | — | — | `skrub.concat([...], axis=1)` — module-level `skrub.concat` does not exist (correct form: `a.skb.concat([b, c], axis=1)`). Revealed after None-placeholder crash was fixed. 3 repairs reverted or re-hallucinated. |
| 08 | Spooky Author | Pass | 1 | Pass | 0.9634 | 0.9635 | Prompt fix alone. |
| 09 | Home Credit | Pass | 1 | Pass | 0.8806 | 0.8806 | Prompt fix alone. |
| 10 | Santander | Pass | 2 | Pass | 0.5730 | 0.5730 | Attempt 1: `skrub.FrequencyEncoder` hallucination (runtime_check caught it). Repaired in 1 round. |

---

## Dynamic Metric Detail (6 passing pipelines)

| # | Metric | Original | Converted | Delta | Tolerance | Pass? |
|---|--------|----------|-----------|-------|-----------|-------|
| 01 | ROC AUC | 0.7684 | 0.7825 | +0.014 | 0.023 | Pass |
| 03 | ROC AUC | 0.8497 | 0.8497 | 0.000 | 0.025 | Pass |
| 04 | R² | 0.6561 | 0.6561 | 0.000 | 0.020 | Pass |
| 08 | ROC AUC macro | 0.9634 | 0.9635 | +0.0001 | 0.029 | Pass |
| 09 | ROC AUC | 0.8806 | 0.8806 | 0.000 | 0.026 | Pass |
| 10 | ROC AUC | 0.5730 | 0.5730 | 0.000 | 0.017 | Pass |

Note: examples 04, 08, 09, 10 use generic var names (`"taxi"`, `"df"`, `"app"`, `"df"`)
instead of the domain-specific names the original harnesses expect. Adapted runners in
`results/run2_harness_{04,08,09,10}.py` pass data under the actual var names.

---

## Remaining Failures — Root Cause Categories

| Root cause | Examples | Why repair loop cannot resolve |
|------------|----------|-------------------------------|
| Hallucinated classes/functions that don't exist in skrub | 02 (`FrequencyEncoder`), 07 (`skrub.concat`) | LLM substitutes other hallucinated names on each repair; no ground-truth replacement is visible in the traceback |
| Function-vs-estimator confusion + stacked errors | 05 | Two simultaneous errors; LLM fixes one per round and breaks the other back |
| Hallucinated prediction-output API for stacking | 06 | `.skb.select(["pred"])` and `.to_pandas()` don't exist on estimator output; no correct DataOps stacking pattern demonstrated in few-shot |

---

## Phase 1 vs Phase 2 (Final)

| | Phase 1 (hand-written) | Phase 2 Run 1 | Phase 2 Run 2 | After Fix 3 |
|---|----------------------|---------------|---------------|-------------|
| Static check pass | 10/10 | 10/10 | 10/10 | 10/10 |
| Runtime check pass | 9/10 | 0/10 | 5/10 | **6/10** |
| Dynamic harness pass | 9/10 | 0/10 | 5/10 | **6/10** |
