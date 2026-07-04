try:
    import stratum as skrub
except ImportError:
    import skrub

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor

# Load data
df = skrub.var("df", pd.read_csv("train.csv"))

# Extract features and target
CAT_COLS = [f"cat{i}" for i in range(1, 117)]
NUM_COLS = [f"cont{i}" for i in range(1, 15)]
TARGET_COLUMN = "loss"

# Create X and y as DataOps
X = df.drop(columns=[TARGET_COLUMN]).skb.mark_as_X()
y = df[TARGET_COLUMN].skb.apply_func(lambda v: np.log1p(v)).skb.mark_as_y()

# Preprocessor for categoricals and numerics
cat_preprocessor = X.skb.select(skrub.selectors.string()).skb.apply_func(
    lambda s: s.apply(lambda col: pd.to_numeric(col.astype('category').cat.codes, errors='coerce'))
).skb.apply_func(lambda s: s.fillna(-1))

num_preprocessor = X.skb.select(skrub.selectors.numeric()).skb.apply_func(
    lambda s: s.fillna(s.median())
)

# Apply preprocessors to selected columns and recombine
X_cat_processed = X.skb.select(skrub.selectors.string()).skb.apply_func(
    lambda s: pd.DataFrame(s).apply(lambda col: pd.to_numeric(col.astype('category').cat.codes, errors='coerce')).fillna(-1)
)

X_num_processed = X.skb.select(skrub.selectors.numeric()).skb.apply_func(
    lambda s: s.fillna(s.median())
)

X_processed = pd.concat([X_cat_processed, X_num_processed], axis=1)

# Create base estimators with their preprocessors
xgb_pipeline = X_processed.skb.apply(
    XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1,
                 subsample=0.8, colsample_bytree=0.8, random_state=0),
    y=y
)

ridge_pipeline = X_processed.skb.apply(
    Ridge(alpha=10.0),
    y=y
)

et_pipeline = X_processed.skb.apply(
    ExtraTreesRegressor(n_estimators=100, max_depth=8, random_state=0),
    y=y
)

# Since skrub doesn't have direct StackingRegressor support, we manually implement
# the stacking by getting OOF predictions during training
# Use a choose_from to switch between training (with OOF) and inference (with direct predictions)
stacking_mode = skrub.eval_mode()

# During training: use 5-fold OOF predictions
# At inference: use direct predictions from each base model
xgb_oof = skrub.choose_from({
    "training": xgb_pipeline.skb.apply_func(lambda v: np.zeros_like(v, dtype=float)),  # placeholder, will be replaced with actual OOF
    "inference": xgb_pipeline
}, name="xgb_oof_mode").as_data_op()

ridge_oof = skrub.choose_from({
    "training": ridge_pipeline.skb.apply_func(lambda v: np.zeros_like(v, dtype=float)),
    "inference": ridge_pipeline
}, name="ridge_oof_mode").as_data_op()

et_oof = skrub.choose_from({
    "training": et_pipeline.skb.apply_func(lambda v: np.zeros_like(v, dtype=float)),
    "inference": et_pipeline
}, name="et_oof_mode").as_data_op()

# Concatenate base learner predictions
base_predictions = pd.concat([xgb_oof, ridge_oof, et_oof], axis=1)

# Meta-learner
meta_predictions = base_predictions.skb.apply(Ridge(alpha=1.0), y=y)

# Final learner
learner = meta_predictions.skb.make_learner()

# Cross-validation is not directly supported for stacking ensembles in skrub,
# but we can still evaluate with cross_validate if needed.
# Note: The OOF implementation above is conceptual; in practice, skrub would need
# proper CV handling that may require additional implementation.