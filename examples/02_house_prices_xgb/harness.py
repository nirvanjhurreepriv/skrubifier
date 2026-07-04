"""
Usage: python harness.py <converted_script_path>
Prints a final JSON line {"original_metric": ..., "converted_metric": ...}
Metric: R² on log1p(SalePrice) scale (the scale the XGBRegressor trains on).
Run inside an environment with skrub, scikit-learn, xgboost, pandas, category_encoders.

NOTE: synthetic train.csv is used (Kaggle House Prices data unavailable); column names
and dtypes match the real dataset so the pipeline runs faithfully end-to-end.

The converted script uses TransformedTargetRegressor(func=log1p, inverse_func=expm1)
so learner.predict() returns original SalePrice; we bring both sides to log scale
for a fair R² comparison (matching what the source pipeline optimises directly).
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
orig_pred_log = PIPELINE.predict(test_df[FEATURE_COLUMNS])
original_metric = r2_score(y_log_test, orig_pred_log)

converted_path = sys.argv[1]
ns = runpy.run_path(converted_path, run_name="__converted__")
learner = ns["learner"]
# Converted learner uses TransformedTargetRegressor; predict() returns original scale.
# Bring to log scale to match original metric.
learner.fit({"houses": train_df})
conv_pred_orig = learner.predict({"houses": test_df})
conv_pred_log = np.log1p(np.maximum(conv_pred_orig, 0))
converted_metric = r2_score(y_log_test, conv_pred_log)

print(json.dumps({"original_metric": original_metric, "converted_metric": converted_metric}))
