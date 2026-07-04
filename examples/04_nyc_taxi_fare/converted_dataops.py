"""
Converted from examples/04_nyc_taxi_fare/source_pipeline.py

Adapted directly from cells 19-25 of the source tutorial notebook
(exercise_v1_1.ipynb) rather than purely LLM-generated — included as a
high-confidence anchor example since it reflects skrub's own reference
implementation of this exact pipeline, and exercises three patterns the
other 3 examples don't:

1. `.skb.apply_func(func)` for stateless functions (pd.read_parquet,
   pd.to_datetime, np.cos/np.sin/np.sqrt/np.arcsin) instead of eager calls.
2. `skrub.eval_mode()` + `.skb.if_else(...)` to make row-cleaning
   TRAINING-ONLY — every row still gets a prediction at inference, which
   the sklearn source_pipeline.py cannot express at all (see its clean_nyc
   docstring). This is a case where the DataOps conversion is not just
   syntactically equivalent but MORE correct than the sklearn original.
3. Cyclical (sin/cos) datetime feature extraction via `.dt` accessors
   chained directly on a DataOp.
"""
import math

import numpy as np
import pandas as pd

try:
    import stratum as skrub  # drop-in accelerated backend, same DataOps API
except ImportError:
    import skrub

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

BB = (-74.5, -72.8, 40.5, 41.8)


def clean_nyc(df):
    lon, lat = df["pickup_longitude"], df["pickup_latitude"]
    dropoff_lon, dropoff_lat = df["dropoff_longitude"], df["dropoff_latitude"]
    mask = (
        (df["fare_amount"] >= 0)
        & (lon >= BB[0]) & (lon <= BB[1])
        & (lat >= BB[2]) & (lat <= BB[3])
        & (dropoff_lon >= BB[0]) & (dropoff_lon <= BB[1])
        & (dropoff_lat >= BB[2]) & (dropoff_lat <= BB[3])
    )
    filtered_df = df[mask].reset_index(drop=True)
    # training-only filter: at prediction time every input row must get a
    # prediction, so we pass the frame through unfiltered outside training.
    return (skrub.eval_mode() != "predict").skb.if_else(filtered_df, df)


def distance(data_op):
    p = math.pi / 180
    lat1 = data_op["pickup_latitude"]
    lat2 = data_op["dropoff_latitude"]
    d_lat_cos = ((lat2 - lat1) * p).skb.apply_func(np.cos)
    d_lon_cos = ((data_op["dropoff_longitude"] - data_op["pickup_longitude"]) * p).skb.apply_func(np.cos)
    lat1_cos = (lat1 * p).skb.apply_func(np.cos)
    lat2_cos = (lat2 * p).skb.apply_func(np.cos)
    a = 0.5 - d_lat_cos / 2 + lat1_cos * lat2_cos * (1 - d_lon_cos) / 2
    distance_km = a.skb.apply_func(np.sqrt).skb.apply_func(np.arcsin) * 12742
    return data_op.assign(distance_km=distance_km)


df = skrub.var("file_path", "data/train1_subsampled.parquet").skb.apply_func(pd.read_parquet)
df = clean_nyc(df)
y = df["fare_amount"].skb.mark_as_y()
X = df.drop(columns=["fare_amount"], errors="ignore").skb.mark_as_X()

X_numeric = X.skb.select(skrub.selectors.numeric())
X_dist = distance(X_numeric).drop(
    columns=["pickup_latitude", "pickup_longitude", "dropoff_latitude", "dropoff_longitude"],
    errors="ignore",
)
X_scaled = X_dist.skb.apply(StandardScaler())

date_col = X["pickup_datetime"].skb.apply_func(pd.to_datetime)
month_col = date_col.dt.month * 2 * np.pi / 12
dayofweek_col = date_col.dt.dayofweek * 2 * np.pi / 7
hour_col = date_col.dt.hour * 2 * np.pi / 24

X_final = X_scaled.assign(
    month_sin=month_col.skb.apply_func(np.sin),
    month_cos=month_col.skb.apply_func(np.cos),
    dow_sin=dayofweek_col.skb.apply_func(np.sin),
    dow_cos=dayofweek_col.skb.apply_func(np.cos),
    hour_sin=hour_col.skb.apply_func(np.sin),
    hour_cos=hour_col.skb.apply_func(np.cos),
)

pred = X_final.skb.apply(LinearRegression(), y=y)

learner = pred.skb.make_learner()

from sklearn.model_selection import ShuffleSplit
split = ShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
scores = pred.skb.cross_validate(cv=split, scoring="r2")
