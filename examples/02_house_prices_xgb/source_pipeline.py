"""Representative of MLE-Bench "house-prices"-style regression pipelines:
log-target regression, XGBoost, mixed numeric/high-cardinality-categorical
columns, target encoding for high-cardinality categoricals."""
import numpy as np
import pandas as pd
from category_encoders import TargetEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

NUM_COLS = ["LotArea", "OverallQual", "YearBuilt", "GrLivArea", "GarageCars"]
LOW_CARD_CAT = ["ExterQual", "KitchenQual"]
HIGH_CARD_CAT = ["Neighborhood"]  # >15 unique values -> target encoded in source solution

TASK = "regression"
FEATURE_COLUMNS = NUM_COLS + LOW_CARD_CAT + HIGH_CARD_CAT
TARGET_COLUMN = "SalePrice"  # source pipeline models log1p(SalePrice)

preprocessor = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), NUM_COLS),
    ("low_card", OneHotEncoder(handle_unknown="ignore"), LOW_CARD_CAT),
    ("high_card", TargetEncoder(), HIGH_CARD_CAT),
])

PIPELINE = Pipeline([
    ("preprocessor", preprocessor),
    ("model", XGBRegressor(n_estimators=800, max_depth=4, learning_rate=0.03, random_state=0)),
])

if __name__ == "__main__":
    df = pd.read_csv("train.csv")
    X, y = df[FEATURE_COLUMNS], np.log1p(df[TARGET_COLUMN])
    PIPELINE.fit(X, y)
