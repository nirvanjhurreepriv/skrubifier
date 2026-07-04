"""
Converted from examples/02_house_prices_xgb/source_pipeline.py

Conversion decisions:
- category_encoders.TargetEncoder for the high-cardinality column is kept
  AS-IS (not silently swapped for skrub.TargetEncoder) to preserve exact
  numerical behavior of the original winning solution; it's passed to
  TableVectorizer's `high_cardinality` kwarg (skrub 0.9+ name; old name was
  `high_cardinality_transformer`).
- log1p(SalePrice) target transform in the source is control flow (a numpy
  call on the target, not a fitted transformer) — lifted into the DAG via
  `.skb.apply_func(np.log1p)` on the target BEFORE mark_as_y(), so the
  XGBRegressor trains on log scale. The learner's .predict() output is
  therefore also in log scale; callers must apply np.expm1 to interpret
  predictions in original dollars.
  NOTE: chaining `.skb.apply_func(np.expm1)` AFTER the estimator node does
  not work in skrub 0.9 — during fit, the estimator node holds the fitted
  estimator object (not predictions), so numpy ufuncs applied to it fail.
  Instead we use sklearn's TransformedTargetRegressor to keep the
  log1p/expm1 transforms encapsulated inside the estimator itself, so
  learner.predict() returns original-scale SalePrice.
"""
import numpy as np
import pandas as pd
try:
    import stratum as skrub  # drop-in accelerated backend, same DataOps API
except ImportError:
    import skrub
from category_encoders import TargetEncoder
from sklearn.compose import TransformedTargetRegressor
from xgboost import XGBRegressor

houses_df = pd.read_csv("train.csv")
houses = skrub.var("houses", houses_df)

X = houses[["LotArea", "OverallQual", "YearBuilt", "GrLivArea", "GarageCars",
            "ExterQual", "KitchenQual", "Neighborhood"]].skb.mark_as_X()
y = houses["SalePrice"].skb.mark_as_y()

# Wrap XGBRegressor with log1p target transform so the estimator trains on
# log scale internally; learner.predict() returns original SalePrice.
model = TransformedTargetRegressor(
    regressor=XGBRegressor(n_estimators=800, max_depth=4, learning_rate=0.03, random_state=0),
    func=np.log1p,
    inverse_func=np.expm1,
)

vectorized = X.skb.apply(
    skrub.TableVectorizer(high_cardinality=TargetEncoder())
)
pred = vectorized.skb.apply(model, y=y)

learner = pred.skb.make_learner()
scores = pred.skb.cross_validate(cv=5, scoring="neg_root_mean_squared_error")
