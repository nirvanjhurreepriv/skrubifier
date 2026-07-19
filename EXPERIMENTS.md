# Experiments

This document describes what "the experiments" are, how to reproduce them
from a clean clone, and how the reported numbers connect to the scripts and
log files.

---

## What the experiments are

The project has two experimental phases.

**Phase 1 — Hand-written reference conversions.**
Each of the 10 source pipelines in `examples/` was manually converted into a
skrub DataOps script (`converted_dataops.py`). A corresponding harness
(`harness.py`) trains both the original and the converted pipeline on the same
held-out split, measures the same metric (ROC AUC or R²), and checks that the
two metrics agree within a tolerance of `max(0.02, 0.03 × |original_metric|)`.
Results are in `results/evaluation_table.md`.

**Phase 2 — Automated LLM conversion (two runs).**
The Skrubifier pipeline (analyzer → LLM → validator + repair loop) was run
on all 10 source pipelines.  The LLM (GWDG SAIA, model `qwen3-coder-next`) was
called via `python -m skrubifier convert` with `--backend gwdg`.  Two full runs
were made, with prompt and framework fixes applied between runs.  The generated
scripts from our canonical Run 2 (plus a targeted Fix 3 patch) are committed
under `results/llm_conversions/`.  Run 1 artifacts are under
`results/llm_conversions_run1/`.  Results are in
`results/phase2_evaluation_table.md`.

---

## How to reproduce

### The single-command path (no API key required)

```bash
git clone <repo-url>
cd skrubifier
python -m venv .venv && source .venv/bin/activate
pip install skrub scikit-learn xgboost pandas numpy pytest openai category_encoders pyarrow
python run_experiments.py
```

`run_experiments.py` executes five steps in order:

| Step | What it does |
|------|-------------|
| 1 | Runs `examples/NN_name/make_data.py` for every example except 04 (which ships committed real data). Each script uses `SEED=42` and writes the synthetic CSV(s) deterministically. |
| 2 | Runs `pytest tests/ -v` (offline unit tests for the Skrubifier framework). |
| 3 | Runs every Phase 1 harness: `examples/NN_name/harness.py converted_dataops.py`. |
| 4 | Re-validates the saved Phase 2 LLM outputs in `results/llm_conversions/` using the adapted harnesses in `results/run2_harness_*.py` and `results/llm_conversions/harnesses/`. |
| 5 | Writes `logs/pytest.log`, `logs/phase1_NN_name.log`, `logs/phase2_NN_name.log`, and a final `logs/summary.md` that compares fresh metrics against the reported tables. |

Expected wall-clock time: 10–20 minutes on a laptop (dominated by the
calibrated XGBoost fit in example 05 and the stacking ensemble in example 06).

The script exits with code 1 if any Phase 1 harness or the pytest suite fails,
so a grader immediately sees breakage.

### What the logs contain

```
logs/
  make_data_01_titanic.log        # stdout/stderr of each make_data.py
  ...
  pytest.log                      # full pytest -v output
  phase1_01_titanic.log           # harness stdout/stderr + JSON result
  ...
  phase2_01_titanic.log           # Phase 2 harness stdout/stderr
  ...
  summary.md                      # table: fresh metrics vs reported values
```

---

## Pointer table

| Experiment | Script | Log file | Reported numbers |
|-----------|--------|----------|-----------------|
| Phase 1: example 01 Titanic | `examples/01_titanic/harness.py` | `logs/phase1_01_titanic.log` | `results/evaluation_table.md` row 01 |
| Phase 1: example 02 House Prices | `examples/02_house_prices_xgb/harness.py` | `logs/phase1_02_house_prices_xgb.log` | `results/evaluation_table.md` row 02 |
| Phase 1: example 03 Credit Fraud | `examples/03_credit_fraud_multitable/harness.py` | `logs/phase1_03_credit_fraud_multitable.log` | `results/evaluation_table.md` row 03 |
| Phase 1: example 04 NYC Taxi | `examples/04_nyc_taxi_fare/harness.py` | `logs/phase1_04_nyc_taxi_fare.log` | `results/evaluation_table.md` row 04 |
| Phase 1: examples 05–10 | `examples/NN_name/harness.py` | `logs/phase1_NN_name.log` | `results/evaluation_table.md` rows 05–10 |
| Phase 2 re-validation: all 10 | `results/run2_harness_*.py` / `results/llm_conversions/harnesses/` | `logs/phase2_NN_name.log` | `results/phase2_evaluation_table.md` |
| Offline unit tests | `pytest tests/` | `logs/pytest.log` | — |
| Project summary (WRITEUP) | `WRITEUP.md` | — | `results/evaluation_table.md`, `results/phase2_evaluation_table.md` |

---

## Reproducibility statement for the LLM generation step

The Phase 2 LLM generation step calls an external language model
(GWDG AcademicCloud SAIA service, model `qwen3-coder-next`) and is **not
bit-reproducible** across re-runs: different random seeds in the model
sampler, network latency, and server-side non-determinism all mean that a
fresh generation run will produce different scripts and may give different
pass/fail counts.

The scripts from our canonical runs are committed under
`results/llm_conversions/` (Run 2 + Fix 3) and `results/llm_conversions_run1/`
(Run 1).  Re-validating these saved scripts — which is what `run_experiments.py`
does in Step 4 — is fully deterministic: the same committed scripts, the same
committed or reproducibly-generated data, and the same fixed random seeds in
all sklearn/XGBoost estimators.

### To re-run LLM generation yourself (requires a GWDG API key)

```bash
export GWDG_API_KEY=<your-key>
# Convert a single example:
python -m skrubifier convert examples/01_titanic/source_pipeline.py \
    --out /tmp/01_converted.py --backend gwdg

# Convert all 10 (writes to results/llm_conversions_rerun/):
for ex in examples/0*/source_pipeline.py; do
    name=$(basename $(dirname $ex))
    python -m skrubifier convert "$ex" \
        --out "results/llm_conversions_rerun/${name}.py" --backend gwdg
done
```

Pass/fail counts may differ slightly from the reported Run 2 results
(6/10 dynamic pass) because the LLM output is non-deterministic.  The
framework fixes (prompts.py, validator.py, analyzer.py) are all committed
and will apply to a fresh generation run.

---

## Cross-model comparison sweep

### What it is

A single-run, three-pipeline probe comparing four open-weight models on the
easy / medium / hard representative pipelines.  Outputs:

| Path | Contents |
|------|----------|
| `results/model_comparison/<model_slug>/0N_name.py` | Generated scripts (reproducible artifact) |
| `results/model_comparison/raw_results.json` | Per-(model, pipeline) metrics and failure causes |
| `results/model_comparison.md` | Summary table, per-pipeline matrix, discussion |
| `results/model_comparison/fig_model_comparison.pdf` | Grouped bar chart |

### Models used (2026-07-19 run)

| Model slug | Full model ID | Type |
|------------|---------------|------|
| `qwen3_coder_next` | `qwen3-coder-next` | code-spec. |
| `openai_gpt_oss_120b` | `openai-gpt-oss-120b` | general |
| `devstral_2_123b_instruct_2512` | `devstral-2-123b-instruct-2512` | code-spec. |
| `glm_4_7` | `glm-4.7` | general |

`llama-3.3-70b-instruct` was not live on the endpoint at run time; devstral
substitutes as a second code-specialist from a different family.

### Pipelines used

| Pipeline | Difficulty |
|----------|-----------|
| `01_titanic` | easy |
| `03_credit_fraud_multitable` | medium |
| `06_allstate_claims_severity` | hard |

### How to re-run (requires GWDG_API_KEY)

```bash
export GWDG_API_KEY=<your-key>
python run_model_comparison.py
```

Total wall-clock time: ~7 minutes (4 models × 3 pipelines, up to 3 repair
rounds each, plus dynamic harness evaluation).

### Reproducibility caveat

Fresh re-generation is **not bit-reproducible** — the LLM output differs
across runs.  The saved scripts in `results/model_comparison/` are the
committed artifact.  To re-validate those saved scripts deterministically
(without calling the LLM), run each saved script through the corresponding
harness directly:

```bash
# Example: re-validate the gpt-oss-120b output for example 01
cd examples/01_titanic
python harness.py ../../results/model_comparison/openai_gpt_oss_120b/01_titanic.py
```
