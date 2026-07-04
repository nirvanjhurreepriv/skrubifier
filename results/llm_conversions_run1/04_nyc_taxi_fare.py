try:
    import stratum as skrub
except ImportError:
    import skrub

import math
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# Define bounding box for NYC
BB = (-74.5, -74.8, 40.5, 41.8)  # corrected to match source: (min_lon, max_lon, min_lat, max_lat)

def clean_nyc(df: pd.DataFrame) -> pd.DataFrame:
    """Training-only row filter: drop bad fares / out-of-NYC coordinates."""
    lon, lat = df["pickup_longitude"], df["pickup_latitude"]
    dlon, dlat = df["dropoff_longitude"], df["dropoff_latitude"]
    target_available = "fare_amount" in df.columns
    mask = (
        (df["fare_amount"] >= 0) if target_available else True
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
    dt = pd.to_datetime(df["pickup_datetime"])
    df["month_sin"] = np.sin(dt.dt.month * 2 * np.pi / 12)
    df["month_cos"] = np.cos(dt.dt.month * 2 * np.pi / 12)
    df["dow_sin"] = np.sin(dt.dt.dayofweek * 2 * np.pi / 7)
    df["dow_cos"] = np.cos(dt.dt.dayofweek * 2 * np.pi / 7)
    df["hour_sin"] = np.sin(dt.dt.hour * 2 * np.pi / 24)
    df["hour_cos"] = np.cos(dt.dt.hour * 2 * np.pi / 24)
    return df.drop(columns=["pickup_datetime"])

# Load data
df = skrub.var("df", pd.read_parquet("data/train1_subsampled.parquet"))

# Mark X and y before transformations
X = df.drop(columns=["fare_amount"]).skb.mark_as_X()
y = df["fare_amount"].skb.mark_as_y()

# Clean rows only during training (skrub-specific correction for train/predict asymmetry)
eval_mode = skrub.eval_mode()
cleaned_X = eval_mode.skb.if_else(
    X.skb.apply_func(lambda df: clean_nyc(df)),
    X  # pass through unchanged at prediction time
)

# Apply feature engineering: distance and date features
# Use apply_func for pure pandas transformations (stateless)
engineered_X = cleaned_X.skb.apply_func(lambda df: add_distance(df))
engineered_X = engineered_X.skb.apply_func(lambda df: add_date_features(df))

# Select the numeric columns for scaling
numeric_cols = ["passenger_count", "distance_km",
                "month_sin", "month_cos", "dow_sin", "dow_cos", "hour_sin", "hour_cos"]
selected_X = engineered_X.skb.select(numeric_cols)

# Apply scaling
scaled_X = selected_X.skb.apply(skrub.StandardScaler())

# Apply the final estimator
pred = scaled_X.skb.apply(LinearRegression(), y=y)

# Create the learner
learner = pred.skb.make_learner()

# If cross-validation was desired (though cv_strategy was null in IR)
# Uncomment the next line to run CV (example for interactive development)
# scores = pred.skb.cross_validate(cv=5, scoring="neg_mean_absolute_error")