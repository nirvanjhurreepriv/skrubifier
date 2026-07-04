"""
Adapted harness for Run 2 example 10 (Santander Customer Transaction).

Run 2 script uses skrub.var("df", df) instead of skrub.var("santander", ...).
This runner passes data under the "df" key.

Usage: python results/run2_harness_10.py  (run from project root)
"""
import json
import os
import runpy
import sys

import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples", "10_santander"))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "examples", "10_santander"))

from source_pipeline import PIPELINE, FEATURE_COLUMNS, TARGET_COLUMN

df = pd.read_csv("train.csv")
train_df, test_df = train_test_split(df, test_size=0.2, random_state=0, stratify=df[TARGET_COLUMN])

PIPELINE.fit(train_df[FEATURE_COLUMNS], train_df[TARGET_COLUMN])
orig_proba = PIPELINE.predict_proba(test_df[FEATURE_COLUMNS])[:, 1]
original_metric = roc_auc_score(test_df[TARGET_COLUMN], orig_proba)

converted_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "llm_conversions", "10_santander.py")
ns = runpy.run_path(converted_path, run_name="__converted__")
learner = ns["learner"]
# Run 2 script uses var name "df" (not "santander" as in original harness)
learner.fit({"df": train_df})
conv_proba = learner.predict_proba({"df": test_df})[:, 1]
converted_metric = roc_auc_score(test_df[TARGET_COLUMN], conv_proba)

print(json.dumps({"original_metric": original_metric, "converted_metric": converted_metric}))
