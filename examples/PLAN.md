# Pipeline selection and planning notes (examples 1–10)

All 10 pipelines are now implemented. This document records the original
selection rationale and planning notes that guided each conversion.
Example #4 is notable: it's built from a real, official skrub DataOps
tutorial notebook (not MLE-Bench/Kaggle), using verified reference syntax
rather than LLM-guessed conversions — see its docstring. It also added two
verified API patterns to `prompts.py` that the earlier examples didn't need:
`.skb.apply_func()` for stateless functions (replacing an earlier,
unverified `skrub.deferred` guess) and `skrub.eval_mode()` +
`.skb.if_else()` for train-only row filtering.

Selection criteria for 5–10: cover the conversion-hazard space, not just easy
cases — picked so each one stresses a different part of
`analyzer.py`/`prompts.py` that examples 1–4 don't.

| # | Source | Pattern | Why it's a good/hard test |
|---|--------|---------|---------------------------|
| 5 | MLE-Bench `random-acts-of-pizza` | text + tabular mix | needs `skrub.TextEncoder`/`StringEncoder` alongside `TableVectorizer`, tests IR's `dtype_hint="text"` path |
| 6 | MLE-Bench `spooky-author-identification` (tabular-adjacent baseline) | leave-one-out target encoding inside CV loop | tests that converter doesn't leak: must express encoding *inside* `.skb.apply`/CV so it refits per fold, not once globally |
| 7 | Kaggle winning-solutions notebook: Otto Group | multi-class, feature engineering via manual `apply(lambda ...)` row-wise | AST fallback path; `custom_function` StepIR with a lambda, no top-level def — analyzer needs a small extension to catch `ast.Lambda` bound to `.apply(...)`, noted as a known gap below |
| 8 | Kaggle winning-solutions notebook: Allstate Claims Severity | `StackingRegressor`-style manual blend of 3 models | no direct skrub primitive; converter must chain 3 `.skb.apply()` branches + concatenate outputs + a meta-learner `.skb.apply()` — good test of `SKRUB_HINTS["StackingClassifier"]`-style guidance |
| 9 | MLE-Bench `home-credit-default-risk` | 6+ auxiliary tables, deep aggregation | stresses `TableIR`/`AggJoiner` chaining at scale; likely needs the IR's `tables` list to carry a join graph, currently only linear chains are modeled — flagged as a **known limitation** (see below) |
| 10 | Kaggle winning-solutions notebook: Santander Customer Transaction | pure-numeric, heavy manual feature selection (`SelectKBest`-style loop) | tests whether `.skb.apply` correctly wraps a `Pipeline` containing a feature-selection step, not just imputers/encoders/model |

## Known limitations to disclose in the evaluation writeup

1. **Join graphs beyond a chain** (pipeline #8): `TableIR` currently models
   `join_to` as a single parent table. Pipelines with a genuine graph
   (products -> baskets, and separately identities -> baskets) need either
   multiple `AggJoiner`/`Joiner` calls chained in sequence (works, just
   verbose) or a small IR extension (`join_to: list[str]`) — worth doing
   before running pipeline 8 through the converter.
2. **Row-wise lambdas** (pipeline #6): `_CustomCodeVisitor` currently only
   captures named `FunctionDef`s and a fixed set of pandas method names; a
   `lambda` passed inline to `.apply()` is invisible to it right now. Fix is
   a ~10 line addition: also visit `ast.Lambda` nodes and attach their
   source to the enclosing `.apply()` call's `StepIR`.
3. **CV-loop-internal fitting** (pipeline #5): the analyzer has no notion of
   "this encoder is refit per fold inside a manual `for train_idx, val_idx
   in kf.split(...)` loop" — it will see it as a `custom_function`. The
   prompt already instructs the LLM to treat CV as `.skb.cross_validate(...)`
   over the *whole* DAG (which naturally refits everything per fold,
   avoiding the leak) — so correctness should hold, but this depends on the
   LLM correctly recognizing the loop as "just CV" rather than something
   more exotic, which is why it's flagged as a target for close manual
   review during evaluation rather than pure automated validation.

## Evaluation protocol for all 10 (matches `validator.py`)

For each pipeline: static check -> dynamic check (fit original + converted
on the same train split, compare held-out metric with tolerance
`max(0.02, 0.03*|original|)`) -> record `{static_ok, dynamic_ok,
original_metric, converted_metric, delta, repair_rounds_needed}` in a
results table. Report: pass rate on first LLM attempt vs. after repair
loop, and qualitative notes on which of the 3 hazard categories above (join
graphs, row-wise lambdas, CV-internal fitting) caused failures.
