"""
Sourced from Anthropic-provided "exercise_v1_1.ipynb" (an official skrub
DataOps tutorial notebook, NYC Taxi Fare Prediction dataset). This file is
the equivalent traditional-sklearn version of the notebook's cell 5 pipeline
(FunctionTransformer + ColumnTransformer + LinearRegression) — reconstructed
from the notebook's own sklearn-baseline cells so the converter has a
concrete Pipeline object to introspect, matching the pattern used by
examples 1-3.

This is the one example in the set where ground truth for the CONVERTED side
is not just plausible but independently verified: cells 19-25 of the
tutorial notebook show Anthropic's/skrub's own DataOps version of this same
pipeline (with row-filtering, haversine distance, and cyclical date
features added incrementally) — see converted_dataops.py, which is adapted
directly from that notebook rather than LLM-generated, and is included here
as a high-confidence few-shot/regression-test anchor for the converter.

Real data included under data/ (train1_subsampled.parquet, test1.parquet,
test1_labels.parquet) — this is the one example that runs fully end-to-end
with real data (both sklearn and skrub sides execute with skrub and pyarrow
installed).
"""
import math

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import FunctionTransformer, Pipeline
from sklearn.preprocessing import StandardScaler

TASK = "regression"
TARGET_COLUMN = "fare_amount"
FEATURE_COLUMNS = [
    "pickup_longitude", "pickup_latitude",
    "dropoff_longitude", "dropoff_latitude",
    "passenger_count", "pickup_datetime",
]
BB = (-74.5, -72.8, 40.5, 41.8)  # NYC bounding box used for row cleaning


def clean_nyc(df: pd.DataFrame) -> pd.DataFrame:
    """Training-only row filter: drop bad fares / out-of-NYC coordinates.
    NOTE: in the sklearn version this is applied identically to train AND
    any data passed through .transform() at inference — this is exactly the
    kind of train/predict asymmetry skrub's eval_mode()+if_else() pattern
    exists to fix; the plain sklearn Pipeline below cannot express "filter
    rows during fit, pass every row through during predict" at all, which is
    itself a correctness gap the converted DataOps version corrects."""
    lon, lat = df["pickup_longitude"], df["pickup_latitude"]
    dlon, dlat = df["dropoff_longitude"], df["dropoff_latitude"]
    mask = (
        (df[TARGET_COLUMN] >= 0) if TARGET_COLUMN in df.columns else True
    ) & (lon.between(BB[0], BB[1])) & (lat.between(BB[2], BB[3])) \
      & (dlon.between(BB[0], BB[1])) & (dlat.between(BB[2], BB[3]))
    return df[mask].reset_index(drop=True)


def add_distance(df: pd.DataFrame) -> pd.DataFrame:
    """Haversine great-circle distance between pickup and dropoff."""
    df = df.copy()
    p = math.pi / 180
    lat1, lat2 = df["pickup_latitude"], df["dropoff_latitude"]
    d_lat = (lat2 - lat1) * p
    d_lon = (df["dropoff_longitude"] - df["pickup_longitude"]) * p
    a = 0.5 - np.cos(d_lat) / 2 + np.cos(lat1 * p) * np.cos(lat2 * p) * (1 - np.cos(d_lon)) / 2
    df["distance_km"] = np.arcsin(np.sqrt(a)) * 12742
    return df.drop(columns=["pickup_latitude", "pickup_longitude",
                             "dropoff_latitude", "dropoff_longitude"])


def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    dt = pd.to_datetime(df["pickup_datetime"])
    df["month_sin"] = np.sin(dt.dt.month * 2 * np.pi / 12)
    df["month_cos"] = np.cos(dt.dt.month * 2 * np.pi / 12)
    df["dow_sin"] = np.sin(dt.dt.dayofweek * 2 * np.pi / 7)
    df["dow_cos"] = np.cos(dt.dt.dayofweek * 2 * np.pi / 7)
    df["hour_sin"] = np.sin(dt.dt.hour * 2 * np.pi / 24)
    df["hour_cos"] = np.cos(dt.dt.hour * 2 * np.pi / 24)
    return df.drop(columns=["pickup_datetime"])


feature_engineering = FunctionTransformer(
    lambda df: add_date_features(add_distance(df))
)
numeric_cols = ["passenger_count", "distance_km",
                 "month_sin", "month_cos", "dow_sin", "dow_cos", "hour_sin", "hour_cos"]
preprocessor = ColumnTransformer([("scale", StandardScaler(), numeric_cols)])

PIPELINE = Pipeline([
    ("feature_engineering", feature_engineering),
    ("preprocessor", preprocessor),
    ("model", LinearRegression()),
])

if __name__ == "__main__":
    df = pd.read_parquet("data/train1_subsampled.parquet")
    df = clean_nyc(df)
    X, y = df[FEATURE_COLUMNS], df[TARGET_COLUMN]
    PIPELINE.fit(X, y)
