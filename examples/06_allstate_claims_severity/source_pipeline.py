"""
Adapted from Kaggle Allstate Claims Severity competition (2016).
The winning/high-scoring solutions used ensembles of gradient boosted trees
and neural nets with log1p(loss) as target.

This file captures the STACKING ENSEMBLE pattern as a representative
simplification: three base regressors (XGBoost, Ridge, ExtraTreesRegressor)
whose OOF predictions are concatenated and fed to a Ridge meta-learner.
The original competition used more complex stacking (50+ base models), but
this 3-model version captures the structural pattern faithfully.

The conversion hazard: manual stacking via ColumnTransformer/Pipeline cannot
be expressed as a single sklearn Pipeline since the base models must predict
on out-of-fold data to avoid leakage. We approximate with a StackingRegressor
(sklearn 0.22+) which handles the CV internals cleanly.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from xgboost import XGBRegressor

CAT_COLS = [f"cat{i}" for i in range(1, 117)]
NUM_COLS = [f"cont{i}" for i in range(1, 15)]
FEATURE_COLUMNS = CAT_COLS + NUM_COLS
TARGET_COLUMN = "loss"
TASK = "regression"

# Encode categoricals (OrdinalEncoder handles string cats robustly with XGBoost/ExtraTrees)
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

preprocessor = ColumnTransformer([
    ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), CAT_COLS),
    ("num", SimpleImputer(strategy="median"), NUM_COLS),
])

base_estimators = [
    ("xgb", Pipeline([
        ("pre", preprocessor),
        ("reg", XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1,
                              subsample=0.8, colsample_bytree=0.8, random_state=0)),
    ])),
    ("ridge", Pipeline([
        ("pre", preprocessor),
        ("reg", Ridge(alpha=10.0)),
    ])),
    ("et", Pipeline([
        ("pre", preprocessor),
        ("reg", ExtraTreesRegressor(n_estimators=100, max_depth=8, random_state=0)),
    ])),
]

PIPELINE = StackingRegressor(
    estimators=base_estimators,
    final_estimator=Ridge(alpha=1.0),
    cv=5,
    passthrough=False,
)

if __name__ == "__main__":
    df = pd.read_csv("train.csv")
    X, y = df[FEATURE_COLUMNS], np.log1p(df[TARGET_COLUMN])
    PIPELINE.fit(X, y)
