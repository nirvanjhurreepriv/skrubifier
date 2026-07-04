"""
Usage: python harness.py <converted_script_path>
Prints a final JSON line {"original_metric": ..., "converted_metric": ...}
Metric: ROC AUC (binary, loan default classification).
Run inside an environment with skrub, scikit-learn, pandas.

NOTE: synthetic application_train.csv + bureau.csv are used (Kaggle Home
Credit data unavailable); column names and data types match the real dataset.

Train/test split is at the application level; bureau records for each split
are filtered accordingly to avoid leakage of test applicant bureau info.
"""
import json
import os
import runpy
import sys

import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier

try:
    import stratum
    if os.environ.get("STRATUM_RUST_BACKEND") == "1":
        stratum.set_config(rust_backend=True, scheduler=True, stats=True)
except ImportError:
    pass

from source_pipeline import build_features, FEATURE_COLUMNS, TARGET_COLUMN

app_df = pd.read_csv("application_train.csv")
bureau_df = pd.read_csv("bureau.csv")

train_app, test_app = train_test_split(
    app_df, test_size=0.2, random_state=0, stratify=app_df[TARGET_COLUMN]
)
train_bureau = bureau_df[bureau_df["SK_ID_CURR"].isin(train_app["SK_ID_CURR"])]
test_bureau  = bureau_df[bureau_df["SK_ID_CURR"].isin(test_app["SK_ID_CURR"])]

train_merged = build_features(train_app, train_bureau)
test_merged  = build_features(test_app,  test_bureau)

model = HistGradientBoostingClassifier(random_state=0)
model.fit(train_merged[FEATURE_COLUMNS], train_merged[TARGET_COLUMN])
orig_proba = model.predict_proba(test_merged[FEATURE_COLUMNS])[:, 1]
original_metric = roc_auc_score(test_merged[TARGET_COLUMN], orig_proba)

converted_path = sys.argv[1]
ns = runpy.run_path(converted_path, run_name="__converted__")
learner = ns["learner"]
learner.fit({"application": train_app, "bureau": train_bureau})
conv_proba = learner.predict_proba({"application": test_app, "bureau": test_bureau})[:, 1]
converted_metric = roc_auc_score(test_app[TARGET_COLUMN], conv_proba)

print(json.dumps({"original_metric": original_metric, "converted_metric": converted_metric}))
