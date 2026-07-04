"""
Usage: python harness.py <converted_script_path>
Prints a final JSON line {"original_metric": ..., "converted_metric": ...}
Metric: ROC AUC macro-averaged over 3 author classes (EAP/HPL/MWS).
Run inside an environment with skrub, scikit-learn, pandas.

NOTE: synthetic train.csv is used (Kaggle Spooky Author Identification data
unavailable); column names match the real dataset exactly.

The source pipeline operates on the raw text Series (TfidfVectorizer takes a
1D array of strings). The converted script wraps the text in a DataFrame to
use DataOps column selection. The harness passes a DataFrame in both cases.
"""
import json
import os
import runpy
import sys

import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize

try:
    import stratum
    if os.environ.get("STRATUM_RUST_BACKEND") == "1":
        stratum.set_config(rust_backend=True, scheduler=True, stats=True)
except ImportError:
    pass

from source_pipeline import PIPELINE, TEXT_COL, TARGET_COLUMN

df = pd.read_csv("train.csv")
train_df, test_df = train_test_split(df, test_size=0.2, random_state=0,
                                     stratify=df[TARGET_COLUMN])

PIPELINE.fit(train_df[TEXT_COL], train_df[TARGET_COLUMN])
classes = PIPELINE.classes_
orig_proba = PIPELINE.predict_proba(test_df[TEXT_COL])
y_bin = label_binarize(test_df[TARGET_COLUMN], classes=classes)
original_metric = roc_auc_score(y_bin, orig_proba, multi_class="ovr", average="macro")

converted_path = sys.argv[1]
ns = runpy.run_path(converted_path, run_name="__converted__")
learner = ns["learner"]
learner.fit({"spooky": train_df})
conv_proba = learner.predict_proba({"spooky": test_df})
converted_metric = roc_auc_score(y_bin, conv_proba, multi_class="ovr", average="macro")

print(json.dumps({"original_metric": original_metric, "converted_metric": converted_metric}))
