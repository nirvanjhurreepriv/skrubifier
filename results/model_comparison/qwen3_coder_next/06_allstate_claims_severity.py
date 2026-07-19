try:
    import stratum as skrub
except ImportError:
    import skrub
import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor

# Load the primary dataset
df = pd.read_csv("train.csv")

# Declare the pipeline input with the loaded DataFrame
data = skrub.var("df", df)

# Define column groups as per the original script
CAT_COLS = [f"cat{i}" for i in range(1, 117)]
NUM_COLS = [f"cont{i}" for i in range(1, 15)]
FEATURE_COLUMNS = CAT_COLS + NUM_COLS
TARGET_COLUMN = "loss"

# Prepare features and target
X = data[FEATURE_COLUMNS].skb.mark_as_X()
y = data[TARGET_COLUMN].skb.apply_func(lambda col: np.log1p(col)).skb.mark_as_y()

# Use TableVectorizer to handle all features (categoricals and numerics)
vectorized = X.skb.apply(skrub.TableVectorizer())

# Create the three base estimators as individual transformations
# Note: Use .fit() on the vectorized data *after* applying the estimators, not before
xgb = vectorized.skb.apply(XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1,
                                       subsample=0.8, colsample_bytree=0.8, random_state=0))

ridge = vectorized.skb.apply(Ridge(alpha=10.0))

et = vectorized.skb.apply(ExtraTreesRegressor(n_estimators=100, max_depth=8, random_state=0))

# Fit all base estimators properly
xgb_fitted = xgb.skb.fit(y=y)
ridge_fitted = ridge.skb.fit(y=y)
et_fitted = et.skb.fit(y=y)

# Build the StackingRegressor with the actual fitted estimators
stacked = StackingRegressor(
    estimators=[
        ('xgb', xgb_fitted._estimator),
        ('ridge', ridge_fitted._estimator),
        ('et', et_fitted._estimator)
    ],
    final_estimator=Ridge(alpha=1.0),
    passthrough=False
)

# Apply the stacked regressor to the vectorized data
final = vectorized.skb.apply(stacked, y=y)

# Export as a SkrubLearner
learner = final.skb.make_learner()

# Optional: run cross-validation to verify the pipeline works
scores = final.skb.cross_validate(cv=5, scoring="neg_mean_squared_error")