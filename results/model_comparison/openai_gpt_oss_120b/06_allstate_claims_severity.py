try:
    import stratum as skrub
except ImportError:
    import skrub
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import ExtraTreesRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor

# ------------------------------------------------------------------
# Load data and declare pipeline inputs
# ------------------------------------------------------------------
df_raw = pd.read_csv("train.csv")
df = skrub.var("df", df_raw)  # primary table

# ------------------------------------------------------------------
# Feature / target definition
# ------------------------------------------------------------------
CAT_COLS = [f"cat{i}" for i in range(1, 117)]
NUM_COLS = [f"cont{i}" for i in range(1, 15)]
FEATURE_COLUMNS = CAT_COLS + NUM_COLS

X = df[FEATURE_COLUMNS].skb.mark_as_X()
y = df["loss"].skb.apply_func(np.log1p).skb.mark_as_y()

# ------------------------------------------------------------------
# Preprocessing shared by all base learners
# ------------------------------------------------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), CAT_COLS),
        ("num", SimpleImputer(strategy="median"), NUM_COLS),
    ]
)

# ------------------------------------------------------------------
# Base estimators (each with its own preprocessing pipeline)
# ------------------------------------------------------------------
base_estimators = [
    (
        "xgb",
        Pipeline(
            steps=[
                ("pre", preprocessor),
                ("reg", XGBRegressor(
                    n_estimators=100,
                    max_depth=5,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=0,
                )),
            ]
        ),
    ),
    (
        "ridge",
        Pipeline(
            steps=[
                ("pre", preprocessor),
                ("reg", Ridge(alpha=10.0)),
            ]
        ),
    ),
    (
        "et",
        Pipeline(
            steps=[
                ("pre", preprocessor),
                ("reg", ExtraTreesRegressor(
                    n_estimators=100,
                    max_depth=8,
                    random_state=0,
                )),
            ]
        ),
    ),
]

# ------------------------------------------------------------------
# Stacking regressor (meta‑learner = Ridge)
# ------------------------------------------------------------------
stacker = StackingRegressor(
    estimators=base_estimators,
    final_estimator=Ridge(alpha=1.0),
    cv=5,
    passthrough=False,
    verbose=0,
)

# ------------------------------------------------------------------
# Apply the model to the design matrix
# ------------------------------------------------------------------
pred = X.skb.apply(stacker, y=y).skb.set_name("stacked_predictions")

# ------------------------------------------------------------------
# Export as a SkrubLearner (ready for training / inference)
# ------------------------------------------------------------------
learner = pred.skb.make_learner()