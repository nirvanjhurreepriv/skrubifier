"""
Adapted harness for Run 2 example 04 (NYC Taxi Fare).

Run 2 script uses skrub.var("taxi", df_raw) instead of the original
skrub.var("file_path", ...).skb.apply_func(pd.read_parquet) pattern.
This runner passes DataFrames directly instead of file paths.

Usage: python results/run2_harness_04.py  (run from project root)
"""
import json
import os
import runpy
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples", "04_nyc_taxi_fare"))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "examples", "04_nyc_taxi_fare"))

from source_pipeline import PIPELINE, FEATURE_COLUMNS, TARGET_COLUMN, clean_nyc

full_df = pd.read_parquet("data/train1_subsampled.parquet")
df = full_df.sample(n=10_000, random_state=0).reset_index(drop=True)
df_clean = clean_nyc(df)
train_df, test_df = train_test_split(df_clean, test_size=0.2, random_state=0)

PIPELINE.fit(train_df[FEATURE_COLUMNS], train_df[TARGET_COLUMN])
orig_pred = PIPELINE.predict(test_df[FEATURE_COLUMNS])
original_metric = r2_score(test_df[TARGET_COLUMN], orig_pred)

converted_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "llm_conversions", "04_nyc_taxi_fare.py")
ns = runpy.run_path(converted_path, run_name="__converted__")
learner = ns["learner"]
# Run 2 script uses var name "taxi" (not "file_path" as in the original pattern)
learner.fit({"taxi": train_df})
conv_pred = learner.predict({"taxi": test_df})

converted_metric = r2_score(test_df[TARGET_COLUMN], conv_pred)
print(json.dumps({"original_metric": original_metric, "converted_metric": converted_metric}))
