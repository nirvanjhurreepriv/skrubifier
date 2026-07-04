"""
Usage: python harness.py <converted_script_path>
Prints a final JSON line {"original_metric": ..., "converted_metric": ...}
Metric: ROC AUC (macro-average, one-vs-rest) across 9 product classes.

NOTE: synthetic train.csv is used (Kaggle Otto Group Product Classification
data is not downloadable in this environment); the 93 anonymous feature columns
and Class_1..Class_9 target labels match the real dataset schema exactly.

XGBoost 3.x requires integer class labels (0..8); both the sklearn pipeline
and the converted DataOps script encode 'Class_N' -> N-1 before fitting.
The harness decodes for the ROC AUC metric.

The original solution averages 44 calibrated GBT configs; source_pipeline.py
captures one representative config as a clean Pipeline. The converted script
replaces the averaging ensemble with skrub.choose_from()-based grid search,
which is the closest DataOps-native equivalent.
"""
import json
import os
import runpy
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, label_binarize

try:
    import stratum
    if os.environ.get("STRATUM_RUST_BACKEND") == "1":
        stratum.set_config(rust_backend=True, scheduler=True, stats=True)
except ImportError:
    pass

from source_pipeline import PIPELINE, FEATURE_COLUMNS, TARGET_COLUMN, CLASSES

df = pd.read_csv("train.csv")
train_df, test_df = train_test_split(df, test_size=0.2, random_state=0,
                                     stratify=df[TARGET_COLUMN])

# Encode 'Class_1'..'Class_9' -> 0..8 for XGBoost 3.x
le = LabelEncoder().fit(CLASSES)
y_train_enc = le.transform(train_df[TARGET_COLUMN])
y_test_enc = le.transform(test_df[TARGET_COLUMN])
y_test_bin = label_binarize(y_test_enc, classes=range(9))

PIPELINE.fit(train_df[FEATURE_COLUMNS], y_train_enc)
orig_proba = PIPELINE.predict_proba(test_df[FEATURE_COLUMNS])
original_metric = roc_auc_score(y_test_bin, orig_proba, multi_class="ovr", average="macro")

converted_path = sys.argv[1]
ns = runpy.run_path(converted_path, run_name="__converted__")
learner = ns["learner"]
learner.fit({"otto": train_df})
conv_proba = learner.predict_proba({"otto": test_df})
# converted learner predicts on integer labels 0..8; decode order matches CLASSES
converted_metric = roc_auc_score(y_test_bin, conv_proba, multi_class="ovr", average="macro")

print(json.dumps({"original_metric": original_metric, "converted_metric": converted_metric}))
