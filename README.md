# Skrubifier

> **Reviewers:** start with [`WRITEUP.md`](WRITEUP.md) — it synthesises the
> whole project (architecture, results, bugs found, limitations) in one place.
>
> **Reproducing results:** run `python run_experiments.py` from the repo root.
> See [`EXPERIMENTS.md`](EXPERIMENTS.md) for the full reproduction guide.

## Repository map

| Path | What it is |
|------|-----------|
| `skrubifier/` | Framework source: `analyzer.py` (sklearn → IR), `converter.py` (LLM call + repair loop), `validator.py` (static + runtime checks), `prompts.py` (API reference + few-shot), `ir.py` (dataclasses), `cli.py` (entry point) |
| `examples/0N_name/` | One pipeline per example: `source_pipeline.py`, `converted_dataops.py`, `harness.py`, `make_data.py` (reproducible data generator, seed=42) |
| `examples/04_nyc_taxi_fare/data/` | Committed real parquet data (NYC Taxi; only real dataset) |
| `results/` | `evaluation_table.md` (Phase 1), `phase2_evaluation_table.md` (Phase 2), LLM-generated scripts in `llm_conversions/` (Run 2) and `llm_conversions_run1/` (Run 1), adapted Phase 2 harnesses |
| `results/model_comparison/` | Cross-model results: `model_comparison.md` (table + matrix + discussion), per-model generated scripts, `fig_model_comparison.pdf` |
| `tests/` | Offline unit tests for the Skrubifier framework (no network, no skrub execution) |
| `run_experiments.py` | Single-command runner: regenerates data → runs tests → runs all harnesses → writes `logs/` |
| `run_model_comparison.py` | Cross-model comparison runner: converts 3 pipelines (easy/medium/hard) across 4 open-weight models, writes `results/model_comparison/` |
| `logs/` | Per-step log files and `summary.md` from the canonical run; committed as submission deliverable |
| `EXPERIMENTS.md` | Full reproduction guide, pointer table, and LLM-step reproducibility statement |
| `CONTRIBUTIONS.md` | Per-member contribution attribution with pointers to specific files |
| `WRITEUP.md` | Project overview, architecture, results, API bugs found, limitations |
| `CHANGELOG.md` | Change log: every non-trivial modification with what, why, how-to-verify |

---

LLM-assisted conversion of tabular ML pipelines (scikit-learn `Pipeline`/
`ColumnTransformer`, XGBoost/LightGBM/CatBoost wrappers, manual pandas
feature-engineering scripts) into the **skrub DataOps** API
(`skrub.var`, `.skb.apply`, `.skb.mark_as_X/y`, `.skb.make_learner`, ...).

## Why this is hard to do with a single LLM prompt

Naively pasting a pipeline into an LLM and asking for "the skrub DataOps
version" fails in practice because:

1. skrub DataOps is a **build-time DAG** (like a lazy computation graph), not
   an eager `fit`/`transform` API — the LLM has to convert *control flow*
   (loops, `if`s, manual joins) into DAG-building calls, not just rename
   functions.
2. Real Kaggle/MLE-Bench solutions mix pandas munging, custom functions,
   multiple tables, CV loops, ensembling and leakage-prone tricks that have
   no 1:1 skrub primitive.
3. Correctness has to be checked *numerically* (do the two pipelines produce
   the same predictions / CV score), not just "does it parse".

Skrubifier addresses this with a **3-stage, tool-assisted pipeline** rather
than a single LLM call:

```
source pipeline (.py / .ipynb)
        │
        ▼
 ┌─────────────────┐   AST + runtime introspection of the sklearn
 │ 1. ANALYZER      │   Pipeline / ColumnTransformer / model objects.
 │  analyzer.py     │   Produces a structured IR (JSON) describing steps,
 │                  │   column groups, estimators, hyperparameters, and
 │                  │   any custom transform code found via AST.
 └─────────────────┘
        │  IR (PipelineIR)
        ▼
 ┌─────────────────┐   LLM (Claude) is prompted with (a) the IR, (b) a
 │ 2. CONVERTER     │   condensed skrub DataOps API reference + few-shot
 │  converter.py    │   examples, (c) the IR->code contract. Emits a
 │                  │   candidate skrub_dataops.py script.
 └─────────────────┘
        │  candidate script
        ▼
 ┌─────────────────┐   Executes original pipeline and candidate script on
 │ 3. VALIDATOR     │   the same held-out split; compares predictions
 │  validator.py    │   (correlation / exact match / metric delta) and
 │                  │   checks the script imports & runs cleanly. Failures
 │                  │   are fed back to the LLM for up to N repair rounds.
 └─────────────────┘
        │
        ▼
  runnable skrub DataOps script + validation report
```

## Package layout

```
skrubifier/
  analyzer.py     # sklearn Pipeline/ColumnTransformer -> PipelineIR (dataclasses)
  ir.py           # IR dataclass definitions (shared contract between stages)
  prompts.py      # System prompt / API reference / few-shot examples for the LLM
  converter.py    # Calls the LLM with the IR, extracts code, orchestrates repair loop
  validator.py    # static_check() + runtime_check() + dynamic_check()
  cli.py          # `python -m skrubifier convert pipeline.py --out out.py`
examples/
  01_titanic/                        # sklearn Pipeline -> skrub DataOps, single table
  02_house_prices_xgb/               # ColumnTransformer + XGBoost, single table
  03_credit_fraud_multitable/        # two tables + groupby aggregation + join
  04_nyc_taxi_fare/                  # real data; date features + geo; eval_mode filter
  05_otto_group/ ... 10_santander/   # examples 05-10: various patterns
  PLAN.md                            # design notes for all 10 pipelines
tests/
  test_analyzer.py
results/
  evaluation_table.md          # Phase 1 dynamic validation results
  phase2_evaluation_table.md   # Phase 2 LLM conversion results
  llm_conversions/             # Phase 2 Run 2 generated scripts
  llm_conversions_run1/        # Phase 2 Run 1 generated scripts (for reference)
  run2_harness_*.py            # Adapted harnesses for Phase 2 validation
logs/
  summary.md                   # Fresh-run metric table (from canonical run_experiments.py)
  *.log                        # Per-step logs
```

## Relationship to stratum

[deem-data/stratum](https://github.com/deem-data/stratum) is **not a
different target syntax** — it's a drop-in accelerated runtime for the same
skrub DataOps operator abstraction (`import stratum as skrub`), adding a
Rust backend, a cost-based optimizer, and a scheduler on top of the
identical `.skb.var/.apply/.mark_as_X/.make_learner` API this framework
already targets. Consequences for this deliverable:

- No separate conversion path is needed. Every script `converter.py`
  produces already targets stratum, since it targets skrub DataOps syntax.
- Generated scripts use `try: import stratum as skrub / except ImportError:
  import skrub` (see `prompts.STRATUM_NOTE`) so the same file runs on
  either, and prefers stratum's Rust backend when installed.
- `validator.dynamic_check(..., use_stratum=True)` sets
  `STRATUM_RUST_BACKEND=1` in the subprocess environment so the dynamic
  validation run picks up stratum automatically if present, without any
  code path divergence from the plain-skrub case.
- Stratum currently has no pip wheel (built from source via
  `maturin develop --release`, requires Rust toolchain + Python 3.12+).
  The "runs on stratum" claim is syntax-level: stratum re-exports the exact
  DataOps operators used across all 10 examples, so every generated script
  is compatible. Execution against the Rust backend has not been verified
  (requires building stratum from source with a Rust toolchain + Python
  3.12+).



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

## LLM backend for the converter

The converter is LLM-agnostic (`llm_call: Callable[[str], str]`). Two
factories are provided:

- **`default_openai_compatible_llm_call()`** — default. Targets GWDG/
  AcademicCloud's SAIA service (`https://chat-ai.academiccloud.de/v1`), a
  free, OpenAI-compatible endpoint serving open-weight models hosted in
  Germany. Get an AcademicCloud ID + SAIA API key via the KISSKI LLM Service
  booking page, then `export GWDG_API_KEY=...` and `pip install openai`.
- **`default_anthropic_llm_call()`** — targets the Anthropic API directly.
  Requires `pip install anthropic` and `ANTHROPIC_API_KEY`.

```bash
python -m skrubifier convert examples/05_x/source_pipeline.py --out out.py            # gwdg (default)
python -m skrubifier convert examples/05_x/source_pipeline.py --out out.py --backend anthropic
```

## Status of this deliverable

- **Framework code complete.** All stages (analyzer, converter, validator,
  repair loop) are implemented; 9/9 tests in `tests/` pass offline (no API
  key or network required). Dynamic/execution validation has been run with
  skrub installed and all harnesses execute.
- **All 10 pipelines done.** Each has a source pipeline, a hand-verified
  converted DataOps script, a validation harness, and (except example 04,
  which ships real data) a seeded `make_data.py` generator. Example 04
  (`04_nyc_taxi_fare`) is adapted directly from an official skrub DataOps
  tutorial with real committed data, serving as a ground-truth regression
  anchor and catching one inaccuracy in the framework's API reference
  (see `prompts.py`'s `.skb.apply_func()` note).
- **Phase 1 (hand-written references): 9/10 pass dynamic validation.**
  Example 06 (Allstate stacking) is a documented structural limitation:
  delta 0.037 vs tolerance 0.020 (see `WRITEUP.md` §7 L1 for details).
- **Phase 2 (automated LLM conversion, GWDG `qwen3-coder-next`): 6/10 pass
  dynamic validation.** The four failures are LLM hallucinations of
  non-existent DataOps API surface; documented in
  `results/phase2_evaluation_table.md`.
- **Cross-model comparison** (4 open-weight models × 3 pipelines) in
  `results/model_comparison/` and `WRITEUP.md` §8.1.
- **Full reproduction:** `python run_experiments.py` regenerates data, runs
  all tests and Phase 1 harnesses, and re-validates saved Phase 2 outputs;
  exits 0 with metrics matching the committed tables.
