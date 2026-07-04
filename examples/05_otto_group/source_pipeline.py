"""
Adapted from cavaunpeu/kaggle-otto (github.com/cavaunpeu/kaggle-otto),
a real solution to Kaggle's Otto Group Product Classification Challenge
(9-class product classification, 93 anonymized numeric features, no
categoricals — a genuinely "pure numeric, single-table" pattern that
diversifies against examples 1-4, which all have categorical columns).

Simplification from the original: the real solution loops over 44
hyperparameter configs (`params_list`), fits a GradientBoostedTrees +
CalibratedClassifierCV pair for each, and averages all 88 sets of
predictions. That loop-and-average ensemble has no direct single-Pipeline
representation, so this file captures ONE representative config as a clean
sklearn Pipeline (the thing the analyzer needs to introspect) and documents
the ensemble intent in a comment — the converted DataOps script expresses
the *search over configs* properly using skrub.choose_from(), which is
actually a more faithful and more inspectable representation of "try many
hyperparameter sets and pick/average the best" than the original's bespoke
Python loop. See PLAN.md notes on pipeline #8 (Allstate) for the same
"no direct skrub primitive for manual ensembling" hazard from a different
angle (stacking rather than averaging).
"""
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

TASK = "classification"
TARGET_COLUMN = "target"
# feat_1 ... feat_93 in the real dataset; kept generic here since the exact
# names don't matter for the analyzer (it only needs the count/dtype).
FEATURE_COLUMNS = [f"feat_{i}" for i in range(1, 94)]

# one representative entry from the original solution's params_list (44 such
# configs were averaged in total).
# NOTE: XGBoost 3.x requires integer class labels (0..8) when using
# multi:softprob; the real Kaggle target is 'Class_1'..'Class_9' (strings).
# The harness and __main__ encode labels to 0..8 before fitting.
gbt = XGBClassifier(
    n_estimators=400,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="multi:softprob",
    num_class=9,
    random_state=0,
)

PIPELINE = Pipeline([
    ("model", CalibratedClassifierCV(gbt, method="isotonic", cv=10)),
])

# Class label encoder: 'Class_1'...'Class_9' -> 0..8
CLASSES = [f"Class_{i}" for i in range(1, 10)]

if __name__ == "__main__":
    from sklearn.preprocessing import LabelEncoder
    df = pd.read_csv("train.csv")  # id, feat_1..feat_93, target ('Class_1'..'Class_9')
    le = LabelEncoder().fit(CLASSES)
    X, y = df[FEATURE_COLUMNS], le.transform(df[TARGET_COLUMN])
    PIPELINE.fit(X, y)