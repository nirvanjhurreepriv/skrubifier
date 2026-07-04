"""
Converted from examples/09_home_credit/source_pipeline.py

Multi-table pattern (application + bureau), same structural hazard as
example 03 but with a different join key (SK_ID_CURR) and more original
application features.

Key conversion decisions:
- AggJoiner was evaluated for the groupby/agg/merge pattern but confirmed
  (see example 03 analysis) to use FITTED (cached training) aggregations at
  predict time, which causes NaN for test applicants not in training bureau
  data. Use `.skb.apply_func()` on the bureau DataOp instead (stateless,
  re-evaluated fresh at each fit/predict call).
- The bureau aggregation mirrors source_pipeline.py's `build_features()`
  exactly: sum of credit, mean debt, count of bureau records, count of
  active credits.
- Categorical encoding: source_pipeline.py encodes categoricals as integer
  codes for HistGBT. TableVectorizer handles this automatically (ordinal /
  low-cardinality encoding for strings), so no explicit OrdinalEncoder needed.
"""
import pandas as pd

try:
    import stratum as skrub  # drop-in accelerated backend, same DataOps API
except ImportError:
    import skrub

from sklearn.ensemble import HistGradientBoostingClassifier

app_df = pd.read_csv("application_train.csv")
bureau_df = pd.read_csv("bureau.csv")

application = skrub.var("application", app_df)
bureau = skrub.var("bureau", bureau_df)

APP_COLS = [
    "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY",
    "DAYS_BIRTH", "DAYS_EMPLOYED",
    "CODE_GENDER", "NAME_EDUCATION_TYPE", "NAME_CONTRACT_TYPE",
]


def _agg_bureau(bureau_df):
    agg = (
        bureau_df.groupby("SK_ID_CURR")
        .agg(
            bureau_credit_sum=("AMT_CREDIT_SUM", "sum"),
            bureau_debt_mean=("AMT_CREDIT_SUM_DEBT", "mean"),
            bureau_count=("SK_ID_BUREAU", "count"),
            bureau_active_count=(
                "CREDIT_ACTIVE", lambda x: (x == "Active").sum()
            ),
        )
        .reset_index()
    )
    return agg


# Aggregate bureau freshly at every fit/predict call (stateless apply_func)
bureau_agg = bureau.skb.apply_func(_agg_bureau)

# Merge with application DataOp; .merge() on a DataOp is tracked as a graph edge
merged = application.merge(bureau_agg, on="SK_ID_CURR", how="left")

FEATURE_COLS = APP_COLS + [
    "bureau_credit_sum", "bureau_debt_mean", "bureau_count", "bureau_active_count"
]
X = merged[FEATURE_COLS].skb.mark_as_X()
y = merged["TARGET"].skb.mark_as_y()

# TableVectorizer handles categorical string columns + numeric + NaN imputation
vectorized = X.skb.apply(skrub.TableVectorizer())
pred = vectorized.skb.apply(HistGradientBoostingClassifier(random_state=0), y=y)

learner = pred.skb.make_learner()
scores = pred.skb.cross_validate(cv=5, scoring="roc_auc")
