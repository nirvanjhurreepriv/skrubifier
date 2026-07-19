# Experiment Run Summary

Run date: 2026-07-19 10:03:06


## Test Suite
pytest tests/ -v : PASS

## Phase 1 — Hand-Written Conversions

| Example | Status | Original | Converted | Delta | Tol | Expected (table) |
|---------|--------|----------|-----------|-------|-----|-----------------|
| 01_titanic | Pass | 0.7190 | 0.7108 | 0.0082 | 0.022 | 0.719 |
| 02_house_prices_xgb | Pass | 0.7841 | 0.7891 | 0.0050 | 0.024 | 0.784 |
| 03_credit_fraud_multitable | Pass | 0.9676 | 0.9676 | 0.0000 | 0.029 | 0.968 |
| 04_nyc_taxi_fare | Pass | 0.6561 | 0.6561 | 0.0000 | 0.020 | 0.656 |
| 05_otto_group | Pass | 0.9660 | 0.9610 | 0.0049 | 0.029 | 0.966 |
| 06_allstate_claims_severity | Fail | 0.5180 | 0.4808 | 0.0372 | 0.020 | 0.518 |
| 07_random_acts_of_pizza | Pass | 0.8053 | 0.8036 | 0.0016 | 0.024 | 0.805 |
| 08_spooky_author | Pass | 0.9613 | 0.9613 | 0.0000 | 0.029 | 0.961 |
| 09_home_credit | Pass | 0.8892 | 0.8839 | 0.0053 | 0.027 | 0.889 |
| 10_santander | Pass | 0.5562 | 0.5562 | 0.0000 | 0.020 | 0.556 |

Phase 1 summary: 9 pass, 1 fail (example 06 is the documented expected failure)

## Phase 2 — Saved LLM Conversions (re-validated, not re-generated)

| Example | Status | Original | Converted | Delta | Expected |
|---------|--------|----------|-----------|-------|----------|
| 01_titanic | Pass | 0.7190 | 0.7108 | 0.0082 | pass |
| 02_house_prices_xgb | Fail (runtime) | — | — | — | fail |
| 03_credit_fraud_multitable | Pass | 0.9676 | 0.9676 | 0.0000 | pass |
| 04_nyc_taxi_fare | Pass | 0.6561 | 0.6561 | 0.0000 | pass |
| 05_otto_group | Fail (runtime) | — | — | — | fail |
| 06_allstate_claims_severity | Fail (runtime) | — | — | — | fail |
| 07_random_acts_of_pizza | Fail (runtime) | — | — | — | fail |
| 08_spooky_author | Pass | 0.9613 | 0.9695 | 0.0082 | pass |
| 09_home_credit | Pass | 0.8892 | 0.8892 | 0.0000 | pass |
| 10_santander | Pass | 0.5562 | 0.5562 | 0.0000 | pass |

Phase 2 summary: 6/10 LLM conversions pass dynamic validation (reported in results/phase2_evaluation_table.md: 6/10)

Note: Phase 2 LLM generation is NOT re-run here.  The saved scripts in
results/llm_conversions/ are validated deterministically against fresh data.
To re-run LLM generation, see EXPERIMENTS.md.
