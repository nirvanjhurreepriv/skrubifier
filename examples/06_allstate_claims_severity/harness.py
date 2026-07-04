"""
Usage: python harness.py <converted_script_path>
Prints a final JSON line {"original_metric": ..., "converted_metric": ...}
Metric: R² on log1p(loss) scale (the scale both the StackingRegressor and
the DataOps meta-learner train on directly).

NOTE: synthetic train.csv is used (Kaggle Allstate Claims Severity data is
not downloadable in this environment); 116 categorical + 14 numeric columns
match the real dataset schema exactly. Loss values are log-normally distributed.

The original source pipeline uses sklearn's StackingRegressor (which computes
OOF base predictions via internal CV); the converted DataOps script uses
in-fold base predictions for the meta-learner. This known semantic difference
means the converted metric may differ slightly more than the usual tolerance —
documented in converted_dataops.py.
"""
import json
import os
import runpy
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

try:
    import stratum
    if os.environ.get("STRATUM_RUST_BACKEND") == "1":
        stratum.set_config(rust_backend=True, scheduler=True, stats=True)
except ImportError:
    pass

from source_pipeline import PIPELINE, FEATURE_COLUMNS, TARGET_COLUMN

df = pd.read_csv("train.csv")
train_df, test_df = train_test_split(df, test_size=0.2, random_state=0)

y_log_train = np.log1p(train_df[TARGET_COLUMN])
y_log_test = np.log1p(test_df[TARGET_COLUMN])

PIPELINE.fit(train_df[FEATURE_COLUMNS], y_log_train)
orig_pred = PIPELINE.predict(test_df[FEATURE_COLUMNS])
original_metric = r2_score(y_log_test, orig_pred)

converted_path = sys.argv[1]
ns = runpy.run_path(converted_path, run_name="__converted__")
learner = ns["learner"]
# Converted learner predicts in log1p scale (same as original).
learner.fit({"allstate": train_df})
conv_pred = learner.predict({"allstate": test_df})
converted_metric = r2_score(y_log_test, conv_pred)

print(json.dumps({"original_metric": original_metric, "converted_metric": converted_metric}))
