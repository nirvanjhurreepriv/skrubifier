"""
Usage: python harness.py <converted_script_path>
Prints a final JSON line {"original_metric": ..., "converted_metric": ...}
Metric: R² on held-out rows.

The converted script uses skrub.var("file_path", ...).skb.apply_func(pd.read_parquet),
so the harness writes temp train/test parquet splits and passes their paths to
the learner.  Both splits retain fare_amount so the clean_nyc DAG node doesn't
fail when it builds the training filter mask (even though eval_mode bypasses
the filter at predict time).

Real data: data/train1_subsampled.parquet (500k rows); subsampled to 10k here
for speed — metrics remain representative.  data/test1.parquet + test1_labels.parquet
hold a separate official test set (also real); NOT used here to keep the
train/test split consistent between original and converted runs.
"""
import json
import os
import runpy
import sys
import tempfile

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

from source_pipeline import PIPELINE, FEATURE_COLUMNS, TARGET_COLUMN, clean_nyc

# Load and subsample the training data for speed
full_df = pd.read_parquet("data/train1_subsampled.parquet")
df = full_df.sample(n=10_000, random_state=0).reset_index(drop=True)
df_clean = clean_nyc(df)

train_df, test_df = train_test_split(df_clean, test_size=0.2, random_state=0)

# Original sklearn pipeline (FunctionTransformer + ColumnTransformer + LinearRegression)
PIPELINE.fit(train_df[FEATURE_COLUMNS], train_df[TARGET_COLUMN])
orig_pred = PIPELINE.predict(test_df[FEATURE_COLUMNS])
original_metric = r2_score(test_df[TARGET_COLUMN], orig_pred)

# Converted DataOps learner: needs file paths, so write temp parquet splits
with tempfile.TemporaryDirectory() as tmpdir:
    train_path = os.path.join(tmpdir, "train_split.parquet")
    test_path = os.path.join(tmpdir, "test_split.parquet")
    train_df.to_parquet(train_path, index=False)
    test_df.to_parquet(test_path, index=False)

    converted_path = sys.argv[1]
    ns = runpy.run_path(converted_path, run_name="__converted__")
    learner = ns["learner"]
    learner.fit({"file_path": train_path})
    conv_pred = learner.predict({"file_path": test_path})

converted_metric = r2_score(test_df[TARGET_COLUMN], conv_pred)

print(json.dumps({"original_metric": original_metric, "converted_metric": converted_metric}))
