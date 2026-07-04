"""
Adapted from Kaggle "Home Credit Default Risk" competition (MLE-Bench).
Binary classification: predict loan default from application + bureau data.

This is a multi-table pipeline: the main loan application table is joined
with aggregated credit bureau data (multiple credit records per applicant).
The original competition had 7+ auxiliary tables; this file uses 2 (application
+ bureau) to keep it runnable while still exercising the AggJoiner pattern.

The conversion challenge: analyzer must detect the manual groupby/agg/merge
pattern and represent it as AggJoiner-compatible TableIR entries. Source
pipeline performs:
  1. Aggregate bureau by SK_ID_CURR (sum of credit, mean debt, count)
  2. Merge onto application table
  3. Impute + HistGradientBoosting

Key differences from example 03 (credit fraud): the join key name is
SK_ID_CURR (not basket_ID), target is 'TARGET', and the main table has
many more original features.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

TARGET_COLUMN = "TARGET"
TASK = "classification"

# Columns from main application table only (after joining bureau aggregates)
APP_COLS = [
    "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY",
    "DAYS_BIRTH", "DAYS_EMPLOYED",
    "CODE_GENDER", "NAME_EDUCATION_TYPE", "NAME_CONTRACT_TYPE",
]

def build_features(app_df: pd.DataFrame, bureau_df: pd.DataFrame) -> pd.DataFrame:
    bureau_agg = (
        bureau_df.groupby("SK_ID_CURR")
        .agg(
            bureau_credit_sum=("AMT_CREDIT_SUM", "sum"),
            bureau_debt_mean=("AMT_CREDIT_SUM_DEBT", "mean"),
            bureau_count=("SK_ID_BUREAU", "count"),
            bureau_active_count=("CREDIT_ACTIVE", lambda x: (x == "Active").sum()),
        )
        .reset_index()
    )
    merged = app_df.merge(bureau_agg, on="SK_ID_CURR", how="left")
    # Encode categoricals as int codes for sklearn (no sklearn OHE here;
    # HistGBT handles NaN but needs numeric)
    for col in ["CODE_GENDER", "NAME_EDUCATION_TYPE", "NAME_CONTRACT_TYPE"]:
        merged[col] = merged[col].astype("category").cat.codes
    return merged

FEATURE_COLUMNS = APP_COLS + [
    "bureau_credit_sum", "bureau_debt_mean", "bureau_count", "bureau_active_count"
]

if __name__ == "__main__":
    app_df = pd.read_csv("application_train.csv")
    bureau_df = pd.read_csv("bureau.csv")
    merged = build_features(app_df, bureau_df)
    X, y = merged[FEATURE_COLUMNS], merged[TARGET_COLUMN]
    model = HistGradientBoostingClassifier(random_state=0)
    model.fit(X, y)
