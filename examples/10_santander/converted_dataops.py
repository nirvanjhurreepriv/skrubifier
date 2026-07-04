"""
Converted from examples/10_santander/source_pipeline.py

Pure-numeric pipeline with feature selection (SelectKBest).

Key conversion decisions:
- Source pipeline: StandardScaler + SelectKBest(f_classif, k=50) + LogisticRegression
  all in a single sklearn Pipeline.
- DataOps equivalent: two `.skb.apply()` calls — one for the scaling+selection
  (wrapped together in a Pipeline inside `.skb.apply()`) and one for the model.
  Alternatively, a single `.skb.apply(sklearn_pipeline)` since sklearn Pipelines
  are valid estimator arguments.
- Chosen approach: wrap the entire sklearn Pipeline (StandardScaler + SelectKBest
  + LogisticRegression) in a single `.skb.apply()`. This is the simplest and
  most faithful representation when there is no branching to express in DataOps.
  DataOps DAG value here is that the data loading and feature/target extraction
  are captured lazily — the actual ML steps are left as a vanilla sklearn Pipeline
  since they have no multi-table or custom-function structure that DataOps improves.
- PLAN.md hazard note: SelectKBest in the IR is kind="transformer" with
  skrub_hint pointing to `.skb.apply(SelectKBest(...))`. This test confirms the
  analyzer correctly identifies it and that the converter can wrap it in a
  `.skb.apply()` rather than trying to replace it with a skrub primitive.
"""
import pandas as pd

try:
    import stratum as skrub  # drop-in accelerated backend, same DataOps API
except ImportError:
    import skrub

from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

df_raw = pd.read_csv("train.csv")
santander = skrub.var("santander", df_raw)

FEATURE_COLUMNS = [f"var_{i}" for i in range(200)]

X = santander[FEATURE_COLUMNS].skb.mark_as_X()
y = santander["target"].skb.mark_as_y()

# Wrap the full sklearn Pipeline as a single DataOps apply step.
# StandardScaler + SelectKBest are sklearn transformers with no DataOps
# equivalent that adds value; wrapping them preserves exact original behavior.
sklearn_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("select", SelectKBest(score_func=f_classif, k=50)),
    ("model", LogisticRegression(C=0.01, max_iter=500, random_state=0)),
])

pred = X.skb.apply(sklearn_pipe, y=y)

learner = pred.skb.make_learner()
scores = pred.skb.cross_validate(cv=5, scoring="roc_auc")
