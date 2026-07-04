"""
Adapted from Kaggle "Santander Customer Transaction Prediction" competition.
Binary classification: predict rare customer transactions (8% positive rate)
from 200 anonymous numeric features (var_0..var_199).

This is the "pure numeric, heavy feature selection" pattern from PLAN.md #10.
The winning solutions found that each feature was independent across classes
(no interactions), so a simple per-feature statistical selection
(SelectKBest/SelectPercentile) followed by a LogisticRegression or
LightGBM was effective.

The conversion hazard: the pipeline wraps a SelectKBest step inside the
sklearn Pipeline. The converted DataOps script must express this as
`.skb.apply(SelectKBest(...))` + `.skb.apply(model)`, or alternatively
collapse both into a single `.skb.apply(Pipeline([select, model]))`.
Key test: does the analyzer correctly identify SelectKBest as a transformer
step (kind="transformer") with an appropriate skrub hint?
"""
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURE_COLUMNS = [f"var_{i}" for i in range(200)]
TARGET_COLUMN = "target"
TASK = "classification"

PIPELINE = Pipeline([
    ("scaler", StandardScaler()),
    ("select", SelectKBest(score_func=f_classif, k=50)),
    ("model", LogisticRegression(C=0.01, max_iter=500, random_state=0)),
])

if __name__ == "__main__":
    df = pd.read_csv("train.csv")
    X, y = df[FEATURE_COLUMNS], df[TARGET_COLUMN]
    PIPELINE.fit(X, y)
