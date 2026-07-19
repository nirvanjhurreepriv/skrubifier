try:
    import stratum as skrub
except ImportError:
    import skrub
import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor

df = pd.read_csv("train.csv")
data = skrub.var("df", df)

CAT_COLS = [f"cat{i}" for i in range(1, 117)]
NUM_COLS = [f"cont{i}" for i in range(1, 15)]
FEATURE_COLUMNS = CAT_COLS + NUM_COLS

X = data[FEATURE_COLUMNS].skb.mark_as_X()
y = data["loss"].skb.mark_as_y()

preprocessor = skrub.TableVectorizer(
    high_cardinality=OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
    numeric_imputer=SimpleImputer(strategy="median")
)

base_estimators = [
    ("xgb", X.skb.apply(preprocessor).skb.apply(
        XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1,
                     subsample=0.8, colsample_bytree=0.8, random_state=0))),
    ("ridge", X.skb.apply(preprocessor).skb.apply(Ridge(alpha=10.0))),
    ("et", X.skb.apply(preprocessor).skb.apply(
        ExtraTreesRegressor(n_estimators=100, max_depth=8, random_state=0)))
]

stacked = skrub.choose_from(base_estimators, name="base_estimators")
meta_features = stacked.skb.concat([est for _, est in base_estimators], axis=1)
pred = meta_features.skb.apply(Ridge(alpha=1.0), y=y)

learner = pred.skb.make_learner()
scores = pred.skb.cross_validate(cv=5, scoring="neg_root_mean_squared_error")