"""
Usage: python harness.py <converted_script_path>
Prints a final JSON line {"original_metric": ..., "converted_metric": ...}
Metric: ROC AUC (binary classification, pizza received or not).
Run inside an environment with skrub, scikit-learn, pandas.

NOTE: synthetic train.csv is used (MLE-Bench random-acts-of-pizza dataset
unavailable for download); column names match the real dataset exactly so
the pipeline runs faithfully end-to-end.

Conversion difference: source uses TfidfVectorizer (sparse n-gram features),
converted uses TableVectorizer -> MinHashEncoder (dense locality-sensitive
hash features). Both are bag-of-words-style encoders; metric gap expected.
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

from source_pipeline import PIPELINE, FEATURE_COLUMNS, TARGET_COLUMN

df = pd.read_csv("train.csv")
train_df, test_df = train_test_split(df, test_size=0.2, random_state=0,
                                     stratify=df[TARGET_COLUMN])

PIPELINE.fit(train_df[FEATURE_COLUMNS], train_df[TARGET_COLUMN])
orig_proba = PIPELINE.predict_proba(test_df[FEATURE_COLUMNS])[:, 1]
original_metric = roc_auc_score(test_df[TARGET_COLUMN], orig_proba)

converted_path = sys.argv[1]
ns = runpy.run_path(converted_path, run_name="__converted__")
learner = ns["learner"]
learner.fit({"pizza": train_df})
conv_proba = learner.predict_proba({"pizza": test_df})[:, 1]
converted_metric = roc_auc_score(test_df[TARGET_COLUMN], conv_proba)

print(json.dumps({"original_metric": original_metric, "converted_metric": converted_metric}))
