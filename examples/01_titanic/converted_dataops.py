"""
Converted from examples/01_titanic/source_pipeline.py

Conversion decisions (what an LLM given the IR should — and does — produce):
- The ColumnTransformer here treats numeric and categorical columns
  differently (impute+scale vs impute+OHE) but skrub.TableVectorizer does
  exactly that split automatically (numeric -> median impute + passthrough/
  scale, low-cardinality categorical -> OneHotEncoder, high-cardinality ->
  a different encoder), so a single TableVectorizer is a faithful, simpler
  replacement rather than manually recreating two skrub selector branches.
- Pclass is numeric in the source CSV but semantically categorical; we keep
  it in the numeric group exactly as the original ColumnTransformer did
  (fidelity to the source > "improving" it) — TableVectorizer will encode it
  as numeric too, matching original behavior.
"""
import pandas as pd
try:
    import stratum as skrub  # drop-in accelerated backend, same DataOps API
except ImportError:
    import skrub
from sklearn.ensemble import RandomForestClassifier

titanic_df = pd.read_csv("train.csv")

titanic = skrub.var("titanic", titanic_df)

X = titanic[["Age", "Fare", "SibSp", "Parch", "Sex", "Embarked", "Pclass"]].skb.mark_as_X()
y = titanic["Survived"].skb.mark_as_y()

vectorized = X.skb.apply(skrub.TableVectorizer())
pred = vectorized.skb.apply(
    RandomForestClassifier(n_estimators=300, max_depth=8, random_state=0), y=y
)

learner = pred.skb.make_learner()

# equivalent of the original's implicit single train/fit — exposed as CV too,
# since MLE-Bench scoring is always over held-out folds:
scores = pred.skb.cross_validate(cv=5, scoring="roc_auc")
