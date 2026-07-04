"""
Usage: python harness.py <converted_script_path>
Prints a final JSON line {"original_metric": ..., "converted_metric": ...}
Metric: ROC AUC on held-out baskets.
Run inside an environment with skrub, scikit-learn, pandas.

NOTE: synthetic baskets.csv/products.csv are used (the real dataset is a
private synthetic credit-fraud-shaped dataset tied to an MLE-Bench competition;
column names and dtypes match the original so pipelines run faithfully).
"""
import json
import os
import runpy
import sys

import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

try:
    import stratum
    if os.environ.get("STRATUM_RUST_BACKEND") == "1":
        stratum.set_config(rust_backend=True, scheduler=True, stats=True)
except ImportError:
    pass

from source_pipeline import build_features, FEATURE_COLUMNS, TARGET_COLUMN

baskets_df = pd.read_csv("baskets.csv")
products_df = pd.read_csv("products.csv")

train_baskets, test_baskets = train_test_split(
    baskets_df, test_size=0.2, random_state=0,
    stratify=baskets_df[TARGET_COLUMN],
)

train_products = products_df[products_df["basket_ID"].isin(train_baskets["basket_ID"])]
test_products = products_df[products_df["basket_ID"].isin(test_baskets["basket_ID"])]

# Original: manual groupby/agg/merge + HistGBT
from sklearn.ensemble import HistGradientBoostingClassifier
train_merged = build_features(train_baskets, train_products)
test_merged = build_features(test_baskets, test_products)
model = HistGradientBoostingClassifier(random_state=0)
model.fit(train_merged[FEATURE_COLUMNS], train_merged[TARGET_COLUMN])
orig_proba = model.predict_proba(test_merged[FEATURE_COLUMNS])[:, 1]
original_metric = roc_auc_score(test_merged[TARGET_COLUMN], orig_proba)

# Converted: DataOps with AggJoiner
converted_path = sys.argv[1]
ns = runpy.run_path(converted_path, run_name="__converted__")
learner = ns["learner"]
learner.fit({"products": train_products, "baskets": train_baskets})
conv_proba = learner.predict_proba({"products": test_products, "baskets": test_baskets})[:, 1]
converted_metric = roc_auc_score(test_baskets[TARGET_COLUMN], conv_proba)

print(json.dumps({"original_metric": original_metric, "converted_metric": converted_metric}))
