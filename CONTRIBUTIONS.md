# Contributions

This file details which parts of the project were contributed by each
member, with pointers to the relevant files. Pipeline selection, the
manual-conversion methodology, and verification of benchmark behavior were
carried out jointly; the attributions below reflect primary authorship and
responsibility for each area.

## Yining Liang

- Manual conversion of pipelines 01, 04, 07, and 10:
  - `examples/01_titanic/` (`source_pipeline.py`, `converted_dataops.py`,
    `harness.py`, `make_data.py`)
  - `examples/04_nyc_taxi_fare/` (all files; the one real-data pipeline)
  - `examples/07_random_acts_of_pizza/` (all files)
  - `examples/10_santander/` (all files)
- Evaluation and result analysis:
  - `results/evaluation_table.md`
  - the evaluation-driven discussion in `WRITEUP.md`

## Charansurya Udaysingh Jhurree

- Manual conversion of pipelines 02, 03, and 08:
  - `examples/02_house_prices_xgb/` (all files)
  - `examples/03_credit_fraud_multitable/` (all files)
  - `examples/08_spooky_author/` (all files)
- Analyzer and intermediate representation development:
  - `skrubifier/analyzer.py`
  - `skrubifier/ir.py`
  - `tests/test_analyzer.py`

## Tiago Frade

- Manual conversion of pipelines 05, 06, and 09:
  - `examples/05_otto_group/` (all files)
  - `examples/06_allstate_claims_severity/` (all files)
  - `examples/09_home_credit/` (all files)
- LLM converter and validation workflow:
  - `skrubifier/converter.py`
  - `skrubifier/validator.py`
  - `skrubifier/prompts.py`
  - `skrubifier/cli.py`
  - Phase 2 automated conversion and its results:
    `results/phase2_evaluation_table.md`,
    `results/phase2_run1_evaluation_table.md`,
    `results/llm_conversions/`, `results/llm_conversions_run1/`

## Shared

- Pipeline selection and diversity planning: `examples/PLAN.md`
- Verification of benchmark behavior across all ten pipelines
- Reproducibility infrastructure: `run_experiments.py`, the per-example
  `make_data.py` scripts, and `logs/`
- Project documentation: `README.md`, `WRITEUP.md`, `CHANGELOG.md`,
  `EXPERIMENTS.md`
