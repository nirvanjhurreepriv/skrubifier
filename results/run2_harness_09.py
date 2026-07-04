"""
Adapted harness for Run 2 example 09 (Home Credit Default Risk).

Run 2 script uses skrub.var("app", app_df) (not "application") but correctly
uses skrub.var("bureau", bureau_df) matching the original. Only the application
var name differs.

Usage: python results/run2_harness_09.py  (run from project root)
"""
import json
import os
import runpy
import sys

import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples", "09_home_credit"))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "examples", "09_home_credit"))

from source_pipeline import build_features, FEATURE_COLUMNS, TARGET_COLUMN

app_df = pd.read_csv("application_train.csv")
bureau_df = pd.read_csv("bureau.csv")

train_app, test_app = train_test_split(app_df, test_size=0.2, random_state=0, stratify=app_df[TARGET_COLUMN])
train_bureau = bureau_df[bureau_df["SK_ID_CURR"].isin(train_app["SK_ID_CURR"])]
test_bureau  = bureau_df[bureau_df["SK_ID_CURR"].isin(test_app["SK_ID_CURR"])]

train_merged = build_features(train_app, train_bureau)
test_merged  = build_features(test_app,  test_bureau)

model = HistGradientBoostingClassifier(random_state=0)
model.fit(train_merged[FEATURE_COLUMNS], train_merged[TARGET_COLUMN])
orig_proba = model.predict_proba(test_merged[FEATURE_COLUMNS])[:, 1]
original_metric = roc_auc_score(test_merged[TARGET_COLUMN], orig_proba)

converted_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "llm_conversions", "09_home_credit.py")
ns = runpy.run_path(converted_path, run_name="__converted__")
learner = ns["learner"]
# Run 2 script uses var name "app" (not "application") and "bureau" (matches)
learner.fit({"app": train_app, "bureau": train_bureau})
conv_proba = learner.predict_proba({"app": test_app, "bureau": test_bureau})[:, 1]
converted_metric = roc_auc_score(test_app[TARGET_COLUMN], conv_proba)

print(json.dumps({"original_metric": original_metric, "converted_metric": converted_metric}))
