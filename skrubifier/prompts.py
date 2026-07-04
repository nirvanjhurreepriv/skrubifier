"""
Prompt construction for stage 2 (converter).

Kept as plain string templates (not an f-string soup in converter.py) so the
API reference / few-shot examples can be updated as skrub's DataOps API
evolves without touching orchestration logic.
"""

STRATUM_NOTE = '''
TARGET RUNTIME: the generated script must run unchanged on both `skrub` and
`stratum` (deem-data/stratum). Stratum is a drop-in accelerated backend for
the exact same DataOps API (`import stratum as skrub`) — it does NOT
introduce different syntax. Therefore:
  - Use `import skrub` and the standard DataOps calls exactly as documented
    below. Do not invent stratum-specific method names.
  - At the top of the script, prefer the pattern:
        try:
            import stratum as skrub
        except ImportError:
            import skrub
    so the same script transparently uses stratum's Rust runtime when
    available and falls back to plain skrub otherwise.
  - Do not call `stratum.set_config(...)` inside the generated pipeline
    logic itself; that is an execution-environment concern the harness/CLI
    sets separately, not part of the DAG definition.
'''

SKRUB_API_REFERENCE = '''
SKRUB DATAOPS — CONDENSED API REFERENCE (do not deviate from these names)
(verified against the official skrub DataOps tutorial notebook — treat this
as ground truth over anything you infer from scikit-learn conventions)

Core building blocks:
  skrub.var("name", dataframe)        -> declare a pipeline input (a DataOp).
      CRITICAL: the second argument is evaluated EAGERLY at script-definition
      time for schema-inference and preview. NEVER pass None or a placeholder.
      ALWAYS pass a real, already-loaded DataFrame:
          df = pd.read_csv("train.csv")
          table = skrub.var("my_table", df)
      Passing None will crash immediately on any subsequent .drop(), .skb.select(),
      .skb.apply(), or any other operation.
  skrub.as_data_op(value)             -> wrap a fixed (non-variable) value as a DataOp
  dataop.skb.mark_as_X()             -> mark a DataOp as the design matrix; ALWAYS
      assign the return value: `X = df[cols].skb.mark_as_X()` (not a void call)
  dataop.skb.mark_as_y()             -> mark a DataOp as the target; ALWAYS assign:
      `y = df[col].skb.mark_as_y()` (not a void call)
  dataop.skb.apply(estimator, y=y)   -> apply a stateful sklearn-compatible estimator/
      transformer (has fit/transform or fit/predict); skrub calls .fit_transform()
      during training and .transform()/.predict() during inference automatically.
  dataop.skb.apply_func(func)        -> call a STATELESS function (no fit/transform
      distinction) on the DataOp's current value, e.g. `df.skb.apply_func(pd.read_parquet)`
      or `col.skb.apply_func(np.cos)`. THIS is the correct way to lift a plain
      function (numpy, pandas top-level funcs, custom helpers) into the DAG —
      prefer it over any other "wrap a function" mechanism.
  dataop.skb.select(selector)        -> select columns, e.g. `X.skb.select(skrub.selectors.numeric())`
  dataop.skb.drop(columns=[...])     -> drop columns from a DataOp (use .skb.drop,
      NOT plain .drop — calling .drop() directly on a DataOp triggers an eager
      preview evaluation that may fail)
  skrub.selectors.numeric() / .string() / .categorical() / .glob(...) / .cols(...)
  dataop.skb.subsample(n)            -> use only n rows during interactive dev/preview
  skrub.eval_mode()                  -> DataOp whose value is "train"/"predict"/"fit_transform"
  dataop.skb.if_else(if_true, if_false) -> branch on a boolean DataOp:
      (skrub.eval_mode() != "predict").skb.if_else(filtered_df, df)
  skrub.choose_from([v1, v2, ...], name="...") or skrub.choose_from({"k": v}, name="...")
      -> discrete hyperparameter/architecture choice (also as estimator param value)
  dataop.skb.concat([other1, other2, ...], axis=1) -> horizontal-stack DataOps.
      USE .skb.concat(), NEVER pd.concat([DataOp, ...]) — pandas concat does not
      accept DataOp objects and will raise a TypeError.
  dataop.skb.make_learner(fitted=False) -> export the DAG as a SkrubLearner
  dataop.skb.cross_validate(cv=..., scoring="...") -> run CV over the whole DAG
  dataop.skb.make_grid_search(cv=..., scoring="...", fitted=True)
  dataop.skb.set_name("...")         -> name a node for readability/reports

Standard pandas ops (+, -, [], .assign, .groupby, .merge, boolean indexing,
.dt.month/.dayofweek/.hour, etc.) work directly on DataOps and are
automatically tracked into the DAG. Call the normal pandas method ON THE
DATAOP itself (e.g. `products.groupby("id").agg(...)` where
`products = skrub.var("products", products_df)`), never on a raw DataFrame
extracted out of it.

Feature-engineering estimators commonly used inside .skb.apply(...):
  skrub.TableVectorizer()            -> auto encode/impute/scale a whole frame
      (subsumes most SimpleImputer+OneHotEncoder+StandardScaler chains)
  skrub.StringEncoder(), skrub.TextEncoder(), skrub.MinHashEncoder(),
  skrub.DatetimeEncoder(), skrub.ToDatetime(), skrub.DropCols()
  skrub.Joiner()                     -> fuzzy/entity join
  skrub.AggJoiner(aux_table=<DataFrame>, aux_key=..., main_key=...,
                  cols=[...], operations=[...])
      -> aggregate aux table and join to main; aux_table MUST be a real
      DataFrame (or the string "X"), NOT a DataOp. To use a second pipeline
      input as the aux table, pass its loaded DataFrame directly:
          products_df = pd.read_csv("products.csv")
          products = skrub.var("products", products_df)
          # pass the underlying df, not the DataOp:
          main.skb.apply(skrub.AggJoiner(aux_table=products_df, ...))
      NOTE: AggJoiner FITS aggregations at training time and CACHES them —
      test entities not in the training aux table get NaN. If the source
      pipeline re-computes aggregations fresh at each call (e.g. by calling
      build_features(test_app, test_bureau) separately), use the stateless
      pattern instead:
          def _agg(aux_df): return aux_df.groupby(...).agg(...).reset_index()
          agg = aux_var.skb.apply_func(_agg)
          merged = main_var.merge(agg, on=key, how="left")

NAMESPACE RULES — avoid common mistakes:
  - StandardScaler, LogisticRegression, XGBRegressor, etc. are SKLEARN
    classes, not skrub classes. Import from sklearn, use via .skb.apply():
        from sklearn.preprocessing import StandardScaler
        scaled = X.skb.apply(StandardScaler())
    NEVER write skrub.StandardScaler() — that class does not exist.
  - skrub.TextEncoder() wraps a sentence-transformers model. It takes no
    TF-IDF parameters (no encoding=, max_features=, stop_words=, etc.).
    For TF-IDF-style text encoding, import TfidfVectorizer from sklearn
    and apply it via .skb.apply().
  - skrub.TableVectorizer(high_cardinality=<encoder>) is the correct
    kwarg name (skrub 0.9+); the old name high_cardinality_transformer
    no longer exists.

RULES FOR THE CONVERTED SCRIPT:
1. Every table the original pipeline reads must become exactly one
   `skrub.var("<name>", <real_loaded_dataframe>)`. Load the DataFrame
   first (e.g., `df = pd.read_csv("train.csv")`), then pass it.
2. Import ALL modules you use at the top of the script (pandas as pd,
   numpy as np, sklearn classes, etc.). Do not use pd.something unless
   `import pandas as pd` appears at the top.
3. Never call sklearn's .fit/.transform directly on a plain DataFrame —
   every transformation must be via .skb.apply / .skb.apply_func.
4. Any plain function (numpy, custom helper, pd.to_datetime, etc.) with
   no fit/transform distinction MUST use `.skb.apply_func(func)`.
5. Any training-only row filter MUST use `skrub.eval_mode() != "predict"`
   + `.skb.if_else(...)`, not a bare conditional.
6. End with `learner = <final_dataop>.skb.make_learner()` and optionally
   `<final_dataop>.skb.cross_validate(...)`.
7. Preserve the original estimator class and hyperparameters exactly.
8. If a step's IR has `source_snippet`, re-implement that EXACT logic as
   DataOp operations.
9. Output ONLY a single fenced python code block. No prose outside.
'''


FEWSHOT_SINGLE_TABLE = '''
--- FEW-SHOT EXAMPLE: single-table classification, ColumnTransformer -> TableVectorizer ---
# ORIGINAL (sklearn)
num_cols = ["Age", "Fare"]
cat_cols = ["Sex", "Embarked"]
pre = ColumnTransformer([
    ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), num_cols),
    ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("ohe", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
])
clf = Pipeline([("pre", pre), ("model", RandomForestClassifier(n_estimators=300, random_state=0))])
clf.fit(X_train, y_train)

# CONVERTED (skrub DataOps) — note: all imports at top, real DataFrame as var default
try:
    import stratum as skrub
except ImportError:
    import skrub
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

titanic_df = pd.read_csv("train.csv")          # load real data; pass it as the var default
df = skrub.var("titanic", titanic_df)          # NEVER use None as the default value

FEATURE_COLS = ["Age", "Fare", "SibSp", "Parch", "Sex", "Embarked", "Pclass"]
X = df[FEATURE_COLS].skb.mark_as_X()           # assign mark_as_X return value — don't discard it
y = df["Survived"].skb.mark_as_y()             # assign mark_as_y return value — don't discard it

vectorized = X.skb.apply(skrub.TableVectorizer())
pred = vectorized.skb.apply(RandomForestClassifier(n_estimators=300, random_state=0), y=y)

learner = pred.skb.make_learner()
scores = pred.skb.cross_validate(cv=5, scoring="roc_auc")
'''

FEWSHOT_MULTI_TABLE = '''
--- FEW-SHOT EXAMPLE: multi-table, stateless groupby+agg+join via apply_func+merge ---
# ORIGINAL (pandas)
agg = products_df.groupby("basket_ID")["cash_price"].agg(["mean", "max"]).reset_index()
merged = baskets_df.merge(agg, on="basket_ID", how="left")
X = merged[["cash_price_mean", "cash_price_max"]].values
y = merged["fraud_flag"]

# CONVERTED (skrub DataOps) — stateless aggregation pattern; re-runs fresh at every call
try:
    import stratum as skrub
except ImportError:
    import skrub
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

baskets_df = pd.read_csv("baskets.csv")        # load real DataFrames before skrub.var()
products_df = pd.read_csv("products.csv")

baskets = skrub.var("baskets", baskets_df)     # NEVER pass None; pass the loaded df
products = skrub.var("products", products_df)


def _agg_products(df):
    return (df.groupby("basket_ID")["cash_price"]
              .agg(["mean", "max"])
              .reset_index()
              .rename(columns={"mean": "cash_price_mean", "max": "cash_price_max"}))


# Use apply_func (stateless) + .merge() for joins that must re-run at predict time.
# This is preferred over AggJoiner when the source pipeline re-computes aggregations
# fresh for each split (build_features called separately on train/test).
agg = products.skb.apply_func(_agg_products)
merged = baskets.merge(agg, on="basket_ID", how="left")

FEATURE_COLS = ["cash_price_mean", "cash_price_max"]
X = merged[FEATURE_COLS].skb.mark_as_X()
y = merged["fraud_flag"].skb.mark_as_y()
vectorized = X.skb.apply(skrub.TableVectorizer())
pred = vectorized.skb.apply(HistGradientBoostingClassifier(), y=y)
learner = pred.skb.make_learner()
'''

REPAIR_TEMPLATE = '''
The candidate script you produced failed validation. Fix it and re-emit the
FULL corrected script (fenced python block only, no prose).

Failure detail:
{failure}

Previous candidate:
{candidate}
'''


def build_conversion_prompt(pipeline_ir_json: str, original_source: str) -> str:
    import json as _json
    try:
        target_col = _json.loads(pipeline_ir_json).get("target_column", "target")
    except Exception:
        target_col = "target"

    target_col_note = (
        f'TARGET COLUMN (mandatory): the target column is exactly "{target_col}". '
        f'Use this exact string when calling .skb.mark_as_y(). '
        f'Do NOT guess it at runtime or search a list of common names like '
        f'["target", "label", "fraud", "is_fraud"] — it is given to you explicitly here.'
    )

    return f'''You are converting a tabular ML pipeline into the skrub DataOps API.

{STRATUM_NOTE}

{SKRUB_API_REFERENCE}

{target_col_note}

{FEWSHOT_SINGLE_TABLE}

{FEWSHOT_MULTI_TABLE}

--- TASK ---
Here is the structured analysis (IR) of the pipeline to convert:

{pipeline_ir_json}

Here is the original source for reference (ground truth for any
custom_function / groupby_agg / join steps in the IR above):

```python
{original_source}
```

Convert this into a single runnable skrub DataOps script following all rules
above. Output only the fenced python code block.
'''
