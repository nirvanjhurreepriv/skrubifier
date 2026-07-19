try:
    import stratum as skrub
except ImportError:
    import skrub

import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

# Load data eagerly for schema inference/purpose
df = pd.read_csv("train.csv")
df_var = skrub.var("df", df)

CAT_COLS = [f"cat{i}" for i in range(1, 117)]
NUM_COLS = [f"cont{i}" for i in range(1, 15)]
FEATURE_COLUMNS = CAT_COLS + NUM_COLS
TARGET_COLUMN = "loss"

X = df_var[FEATURE_COLUMNS].skb.mark_as_X()
y = df_var[TARGET_COLUMN].skb.mark_as_y()

# Define the shared preprocessor for the base learners.
# In StackingRegressor, this preprocessor is fitted to the training data
# and reused during folding and prediction, matching the requirements.
preprocessor = ColumnTransformer(
    [
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), CAT_COLS),
        ("num", SimpleImputer(strategy="median"), NUM_COLS),
    ]
)

# Build the StackingRegressor.
# Note: In the original source, `y` is transformed with `np.log1p` before fit.
# To faithfully replicate the pipeline, we apply this log1p transformation to y
# passed to the model via .skb.apply.
y_transformed = y.skb.apply_func(np.log1p)

stacking_regressor = StackingRegressor(
    estimators=[
        ("xgb", Pipeline([("pre", preprocessor), ("reg", XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8, random_state=0))])),
        ("ridge", Pipeline([("pre", preprocessor), ("reg", Ridge(alpha=10.0))])),
        ("et", Pipeline([("pre", preprocessor), ("reg", ExtraTreesRegressor(n_estimators=100, max_depth=8, random_state=0))])),
    ],
    final_estimator=Ridge(alpha=1.0),
    cv=5,
    passthrough=False,
)

# Apply the stacking regressor. The predictions will be in the log1p(loss) space.
pred = X.skb.apply(stacking_regressor, y=y_transformed)

learner = pred.skb.make_learner()