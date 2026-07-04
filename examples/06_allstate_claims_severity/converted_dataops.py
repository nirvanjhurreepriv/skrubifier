"""
Converted from examples/06_allstate_claims_severity/source_pipeline.py

Conversion of the stacking ensemble (StackingRegressor -> DataOps branches):
- No direct skrub primitive for StackingRegressor.
- Three `.skb.apply()` branches (XGBoost, Ridge, ExtraTrees) each use their
  own TableVectorizer instance over the full feature set.
- Their Series predictions are wrapped in DataFrames via `.skb.apply_func()`
  (skb.concat requires DataFrame inputs, not Series) then column-concatenated
  with `.skb.concat([...], axis=1)`.
- A Ridge meta-learner is applied on the stacked DataOp.
- Target: log1p(loss) via `.skb.apply_func(np.log1p)` BEFORE `mark_as_y()`
  (stateless transform captured as a graph node). All models train/predict in
  log1p scale; `learner.predict()` also returns log1p scale — the harness
  applies expm1 separately if original-scale predictions are needed.

Note on leakage vs sklearn StackingRegressor:
  sklearn's StackingRegressor computes OOF predictions via internal CV so the
  meta-learner never sees training-set predictions. The DataOps version
  applies all three branches on the same training data -> mild leakage for the
  meta-learner. This is a documented limitation; a fully-clean DataOps version
  would need `skb.cross_validate`-style mechanics that are more complex and
  not documented in SKRUB_API_REFERENCE. For metric comparison on a held-out
  test split, both approaches still produce valid (if differently-calibrated)
  predictions.
"""
import numpy as np
import pandas as pd

try:
    import stratum as skrub  # drop-in accelerated backend, same DataOps API
except ImportError:
    import skrub

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor

df_raw = pd.read_csv("train.csv")
data = skrub.var("allstate", df_raw)

CAT_COLS = [f"cat{i}" for i in range(1, 117)]
NUM_COLS = [f"cont{i}" for i in range(1, 15)]
FEATURE_COLUMNS = CAT_COLS + NUM_COLS

X = data[FEATURE_COLUMNS].skb.mark_as_X()
# log1p on target is stateless: apply_func BEFORE mark_as_y
y = data["loss"].skb.apply_func(np.log1p).skb.mark_as_y()

# Branch 1: XGBoost
x_vec1 = X.skb.apply(skrub.TableVectorizer())
pred_xgb = x_vec1.skb.apply(
    XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1,
                 subsample=0.8, colsample_bytree=0.8, random_state=0),
    y=y,
)

# Branch 2: Ridge
x_vec2 = X.skb.apply(skrub.TableVectorizer())
pred_ridge = x_vec2.skb.apply(Ridge(alpha=10.0), y=y)

# Branch 3: ExtraTrees
x_vec3 = X.skb.apply(skrub.TableVectorizer())
pred_et = x_vec3.skb.apply(
    ExtraTreesRegressor(n_estimators=100, max_depth=8, random_state=0),
    y=y,
)

# Each branch outputs a Series; wrap in a named DataFrame for horizontal concat
# (skb.concat requires DataFrame inputs, not Series)
pred_xgb_df = pred_xgb.skb.apply_func(lambda s: pd.DataFrame({"pred_xgb": s}))
pred_ridge_df = pred_ridge.skb.apply_func(lambda s: pd.DataFrame({"pred_ridge": s}))
pred_et_df = pred_et.skb.apply_func(lambda s: pd.DataFrame({"pred_et": s}))

# Column-stack the three base predictions
stacked = pred_xgb_df.skb.concat([pred_xgb_df, pred_ridge_df, pred_et_df], axis=1)

# Meta-learner: Ridge on stacked log1p-scale predictions; output is also log1p-scale
pred_final = stacked.skb.apply(Ridge(alpha=1.0), y=y)

learner = pred_final.skb.make_learner()
scores = pred_final.skb.cross_validate(cv=5, scoring="neg_root_mean_squared_error")
