try:
    import stratum as skrub
except ImportError:
    import skrub

import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor

# Load data
df = pd.read_csv("train.csv")

# Define column groups
CAT_COLS = [f"cat{i}" for i in range(1, 117)]
NUM_COLS = [f"cont{i}" for i in range(1, 15)]
FEATURE_COLUMNS = CAT_COLS + NUM_COLS
TARGET_COLUMN = "loss"

# Declare input DataOp with real DataFrame (critical: not None)
data = skrub.var("df", df)

# Prepare features and target
X = data[FEATURE_COLUMNS].skb.mark_as_X()
y = data[TARGET_COLUMN].skb.apply_func(np.log1p).skb.mark_as_y()

# Apply TableVectorizer to handle both numeric and categorical columns automatically
vectorized = X.skb.apply(skrub.TableVectorizer())

# Base estimators: apply each on the same vectorized features
xgb_pred = vectorized.skb.apply(
    XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1,
                 subsample=0.8, colsample_bytree=0.8, random_state=0),
    y=y
)

ridge_pred = vectorized.skb.apply(
    Ridge(alpha=10.0),
    y=y
)

et_pred = vectorized.skb.apply(
    ExtraTreesRegressor(n_estimators=100, max_depth=8, random_state=0),
    y=y
)

# Concatenate predictions horizontally for the meta-learner
# Use pandas.concat via .to_pandas() for horizontal concatenation
# Fix: Extract the 'pred' column as a DataFrame, not Series
xgb_df = xgb_pred.skb.select(["pred"]).to_pandas()
ridge_df = ridge_pred.skb.select(["pred"]).to_pandas()
et_df = et_pred.skb.select(["pred"]).to_pandas()

# Ensure we have DataFrames (not Series) by keeping column structure
stacked = pd.concat([xgb_df, ridge_df, et_df], axis=1)
stacked.columns = ["xgb_pred", "ridge_pred", "et_pred"]

# Convert back to DataOp for skrub pipeline compatibility
stacked = skrub.var("stacked", stacked)

# Meta-learner: Ridge regressor on the stacked predictions
meta_pred = stacked.skb.apply(Ridge(alpha=1.0), y=y)

# Export as a learner for training/inference
learner = meta_pred.skb.make_learner()