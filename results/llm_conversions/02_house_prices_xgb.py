import skrub
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.preprocessing import OneHotEncoder

# Load real DataFrame first, then pass as var default
df = pd.read_csv("train.csv")
data = skrub.var("data", df)

# Define column groups per the IR
NUM_COLS = ["LotArea", "OverallQual", "YearBuilt", "GrLivArea", "GarageCars"]
LOW_CARD_CAT = ["ExterQual", "KitchenQual"]
HIGH_CARD_CAT = ["Neighborhood"]
FEATURE_COLUMNS = NUM_COLS + LOW_CARD_CAT + HIGH_CARD_CAT

# Extract X and y
X = data[FEATURE_COLUMNS].skb.mark_as_X()
# Apply log1p to target as in the original source
y = data["SalePrice"].skb.apply_func(np.log1p).skb.mark_as_y()

# Use TableVectorizer with appropriate encoders for each column group
# - Numeric: impute median + standard scaling is default for numeric columns in TableVectorizer
# - Low cardinality: OneHotEncoder via low_cardinality_transformer
# - High cardinality: FrequencyEncoder (replacement for TargetEncoder in newer skrub versions)
vectorized = X.skb.apply(
    skrub.TableVectorizer(
        low_cardinality_transformer=OneHotEncoder(handle_unknown="ignore"),
        high_cardinality_transformer=skrub.FrequencyEncoder(min_frequency=20)
    ),
    y=y
)

# Apply the estimator
pred = vectorized.skb.apply(
    XGBRegressor(
        n_estimators=800,
        max_depth=4,
        learning_rate=0.03,
        random_state=0,
        objective="reg:squarederror"
    ),
    y=y
)

# Export as learner
learner = pred.skb.make_learner()