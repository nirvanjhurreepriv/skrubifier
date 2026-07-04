"""
Adapted harness for Run 2 example 08 (Spooky Author Identification).

Run 2 script uses skrub.var("df", df) instead of skrub.var("spooky", ...).
This runner passes data under the "df" key.

Usage: python results/run2_harness_08.py  (run from project root)
"""
import json
import os
import runpy
import sys

import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples", "08_spooky_author"))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "examples", "08_spooky_author"))

from source_pipeline import PIPELINE, TEXT_COL, TARGET_COLUMN

df = pd.read_csv("train.csv")
train_df, test_df = train_test_split(df, test_size=0.2, random_state=0, stratify=df[TARGET_COLUMN])

PIPELINE.fit(train_df[TEXT_COL], train_df[TARGET_COLUMN])
classes = PIPELINE.classes_
orig_proba = PIPELINE.predict_proba(test_df[TEXT_COL])
y_bin = label_binarize(test_df[TARGET_COLUMN], classes=classes)
original_metric = roc_auc_score(y_bin, orig_proba, multi_class="ovr", average="macro")

converted_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "llm_conversions", "08_spooky_author.py")
ns = runpy.run_path(converted_path, run_name="__converted__")
learner = ns["learner"]
# Run 2 script uses var name "df" (not "spooky" as in original harness)
learner.fit({"df": train_df})
conv_proba = learner.predict_proba({"df": test_df})
converted_metric = roc_auc_score(y_bin, conv_proba, multi_class="ovr", average="macro")

print(json.dumps({"original_metric": original_metric, "converted_metric": converted_metric}))
