try:
    import stratum as skrub
except ImportError:
    import skrub
import pandas as pd
import numpy as np
import math
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# Define constants
BB = (-74.5, -72.8, 40.5, 41.8)  # NYC bounding box

def clean_nyc(df: pd.DataFrame) -> pd.DataFrame:
    """Training-only row filter: drop bad fares / out-of-NYC coordinates."""
    lon = df["pickup_longitude"]
    lat = df["pickup_latitude"]
    dlon = df["dropoff_longitude"]
    dlat = df["dropoff_latitude"]
    target_col = "fare_amount"
    
    # Only filter on fare_amount if it exists (training data)
    fare_mask = (df[target_col] >= 0) if target_col in df.columns else True
    mask = (
        fare_mask &
        lon.between(BB[0], BB[1]) &
        lat.between(BB[2], BB[3]) &
        dlon.between(BB[0], BB[1]) &
        dlat.between(BB[2], BB[3])
    )
    return df[mask].reset_index(drop=True)

def add_distance(df: pd.DataFrame) -> pd.DataFrame:
    """Haversine great-circle distance between pickup and dropoff."""
    p = math.pi / 180
    lat1 = df["pickup_latitude"]
    lat2 = df["dropoff_latitude"]
    d_lat = (lat2 - lat1) * p
    d_lon = (df["dropoff_longitude"] - df["pickup_longitude"]) * p
    a = 0.5 - np.cos(d_lat) / 2 + np.cos(lat1 * p) * np.cos(lat2 * p) * (1 - np.cos(d_lon)) / 2
    df = df.copy()
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

# Load data
df_raw = pd.read_parquet("data/train1_subsampled.parquet")

# Declare input variable (eager evaluation of real DataFrame)
df = skrub.var("taxi", df_raw)

# Apply training-only row filter using eval_mode and if_else
cleaned_df = (skrub.eval_mode() != "predict").skb.if_else(
    df.skb.apply_func(clean_nyc),
    df
)

# Feature engineering: add distance and date features
engineered_df = cleaned_df.skb.apply_func(add_distance).skb.apply_func(add_date_features)

# Select target and feature columns
target_col = "fare_amount"
feature_cols = [
    "passenger_count", "distance_km",
    "month_sin", "month_cos", "dow_sin", "dow_cos", "hour_sin", "hour_cos"
]

# Mark target and design matrix
X = engineered_df[feature_cols].skb.mark_as_X()
y = engineered_df[target_col].skb.mark_as_y()

# Apply scaling to numeric features
scaled_X = X.skb.apply(StandardScaler())

# Apply LinearRegression estimator
pred = scaled_X.skb.apply(LinearRegression(), y=y)

# Create learner and run cross-validation
learner = pred.skb.make_learner()
cv_scores = pred.skb.cross_validate(cv=5, scoring="neg_mean_absolute_error")