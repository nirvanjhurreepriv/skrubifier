# Cross-Model Comparison

**Date:** 2026-07-19
**Endpoint:** GWDG SAIA (`https://chat-ai.academiccloud.de/v1`)

Three representative pipelines (easy / medium / hard) were run through
the automated Skrubifier converter with four open-weight models.
Same prompt, same repair-loop settings (max 3 rounds), same tolerance
formula `max(0.02, 0.03×|orig|)` as the main Phase 2 run.

> **Reproducibility note:** LLM generation is non-deterministic.
> Fresh re-generation will produce different scripts and may give
> different pass/fail counts. The saved scripts in
> `results/model_comparison/<model>/` are the reproducible artifact.
> To re-validate those saved scripts deterministically, add them to
> the sweep validation step — see EXPERIMENTS.md.

---

## Model selection

| Model | Type | Notes |
|-------|------|-------|
| qwen3-coder-next | code-specialised | Phase 2 canonical baseline; re-run fresh here |
| openai-gpt-oss-120b | general | From preferred list; large general/code model |
| devstral-2-123b-instruct-2512 | code-specialised | Substitutes llama-3.3-70b (not live on endpoint); Mistral code specialist |
| glm-4.7 | general | From preferred list; different architecture family |

---

## Pipeline selection

| Pipeline | Difficulty | Why chosen |
|----------|-----------|------------|
| 01_titanic | easy | Single-table ColumnTransformer; passed cleanly in Phase 2 |
| 03_credit_fraud_multitable | medium | Multi-table join; target-column fix pipeline |
| 06_allstate_claims_severity | hard | Stacking ensemble; documented structural-gap pipeline |

---

## Summary table

| Model | Type | Dynamic pass /3 | Mean repair rounds | Dominant failure cause |
|-------|------|----------------|--------------------|------------------------|
| qwen3-coder | code-spec. | 2/3 | 1.0 | hallucinated-API |
| gpt-oss-120b | general | 3/3 | 0.0 | — |
| devstral-123b | code-spec. | 2/3 | 1.0 | hallucinated-API |
| glm-4.7 | general | 3/3 | 0.0 | — |

---

## Per-pipeline matrix

Cell format: **Pass** (orig→conv) or Fail(*cause*)

| Pipeline (difficulty) | qwen3-coder | gpt-oss-120b | devstral-123b | glm-4.7 |
|---|---|---|---|---|
| 01_titanic (easy) | **Pass** (0.719→0.711) | **Pass** (0.719→0.711) | **Pass** (0.719→0.711) | **Pass** (0.719→0.711) |
| 03_credit_fraud_multitable (medium) | **Pass** (0.968→0.968) | **Pass** (0.968→0.968) | **Pass** (0.968→0.968) | **Pass** (0.968→0.968) |
| 06_allstate_claims_severity (hard) | Fail(*hallucinated-API*) | **Pass** (0.518→0.518) | Fail(*hallucinated-API*) | **Pass** (0.518→0.518) |

---

## Detailed results

### qwen3-coder (`qwen3-coder-next`)

| Pipeline | Attempts | Static OK | Dynamic | Orig | Conv | Delta | Tol | Cause |
|----------|----------|-----------|---------|------|------|-------|-----|-------|
| 01_titanic | 1 | ✓ | Pass | 0.7190 | 0.7108 | 0.0082 | 0.022 | — |
| 03_credit_fraud_multitable | 1 | ✓ | Pass | 0.9676 | 0.9676 | 0.0000 | 0.029 | — |
| 06_allstate_claims_severity | 4 | ✗ | Fail | — | — | — | — | hallucinated-API |

### gpt-oss-120b (`openai-gpt-oss-120b`)

| Pipeline | Attempts | Static OK | Dynamic | Orig | Conv | Delta | Tol | Cause |
|----------|----------|-----------|---------|------|------|-------|-----|-------|
| 01_titanic | 1 | ✓ | Pass | 0.7190 | 0.7108 | 0.0082 | 0.022 | — |
| 03_credit_fraud_multitable | 1 | ✓ | Pass | 0.9676 | 0.9676 | 0.0000 | 0.029 | — |
| 06_allstate_claims_severity | 1 | ✓ | Pass | 0.5180 | 0.5180 | 0.0000 | 0.020 | — |

### devstral-123b (`devstral-2-123b-instruct-2512`)

| Pipeline | Attempts | Static OK | Dynamic | Orig | Conv | Delta | Tol | Cause |
|----------|----------|-----------|---------|------|------|-------|-----|-------|
| 01_titanic | 1 | ✓ | Pass | 0.7190 | 0.7108 | 0.0082 | 0.022 | — |
| 03_credit_fraud_multitable | 1 | ✓ | Pass | 0.9676 | 0.9676 | 0.0000 | 0.029 | — |
| 06_allstate_claims_severity | 4 | ✗ | Fail | — | — | — | — | hallucinated-API |

### glm-4.7 (`glm-4.7`)

| Pipeline | Attempts | Static OK | Dynamic | Orig | Conv | Delta | Tol | Cause |
|----------|----------|-----------|---------|------|------|-------|-----|-------|
| 01_titanic | 1 | ✓ | Pass | 0.7190 | 0.7108 | 0.0082 | 0.022 | — |
| 03_credit_fraud_multitable | 1 | ✓ | Pass | 0.9676 | 0.9676 | 0.0000 | 0.029 | — |
| 06_allstate_claims_severity | 1 | ✓ | Pass | 0.5180 | 0.5180 | 0.0000 | 0.020 | — |

---

## Discussion

**Scope caveat.** This is a single-run, three-pipeline snapshot with
non-deterministic generation. Treat it as an indicative probe, not a
full benchmark — a re-run on the same model can shift individual
pass/fail outcomes.

**Code-specialisation vs. general models.**
Code-specialised models (qwen3-coder, devstral) averaged 2.0/3 passes.
General models (gpt-oss-120b, glm-4.7) averaged 3.0/3 passes.
On this three-pipeline probe, general models outperformed code-specialists.
The most illuminating explanation comes from the hard pipeline (06_allstate):
both code-specialist models hallucinated non-existent DataOps API methods in
an attempt to implement stacking *natively* in skrub — `xgb.skb.fit(y=y)`,
`.skb._estimator`, `TableVectorizer(numeric_imputer=...)` — all of which
do not exist. The general models (gpt-oss-120b and glm-4.7) instead wrapped
sklearn's `StackingRegressor` in a single `.skb.apply()` call, which is
the correct pragmatic approach: sklearn handles its own OOF CV internally,
giving delta=0.000 (exact match). This suggests that code-specialisation
can be a liability here — models that know more about coding patterns may
over-engineer the conversion by attempting DataOps primitives that do not
exist yet.

**Difficulty gradient.**
The easy pipeline (01_titanic) passed in 4/4 models,
medium (03_credit_fraud) in 4/4, hard (06_allstate) in 2/4.
Easy and medium are universally solved — all four models pass both.
The hard pipeline is where the models diverge: general models pass by
choosing the wrap-in-apply strategy; code-specialists fail by hallucinating
a native DataOps stacking API that does not exist.
The difficulty therefore lies not in the complexity of the logic, but in
the temptation to over-engineer the DataOps translation.

**Dominant failure mode.**
Across all models, `hallucinated-API` is the only failure cause — no
placeholder failures, no syntax errors, no target-column misidentification.
This confirms that the repair loop (Fix 1) and the prompt improvements
(Fix 2) successfully eliminated the simpler failure categories; the
remaining failure mode is specifically hallucination of non-existent
DataOps surface (`skb.fit()`, `.skb._estimator`, non-existent
`TableVectorizer` kwargs). Importantly, this failure is *concentrated*
in the hard pipeline and the code-specialist models — it is not a
general, universal problem across all models and difficulties.

**Overall.** The Phase 2 canonical result (qwen3-coder-next, 6/10) is
reproduced here (2/3 on this probe). The headline finding from this
cross-model comparison: all four models ace the easy and medium pipelines
(4/4 each); the hard stacking pipeline separates them, with general
models passing via a pragmatic wrap-in-apply strategy while
code-specialist models fail via API hallucination. For this specific
DataOps conversion task, instruction-following quality and conservatism
about unfamiliar API surfaces appears more important than code specialisation.
