try:
    import stratum as skrub
except ImportError:
    import skrub
import numpy as np
from xgboost import XGBRegressor

# Load primary table
df = skrub.var("df", pd.read_csv("train.csv"))

# Extract features and target (apply log1p transformation to target)
X = df[["LotArea", "OverallQual", "YearBuilt", "GrLivArea", "GarageCars", "ExterQual", "KitchenQual", "Neighborhood"]]
y = df["SalePrice"].skb.apply_func(np.log1p).skb.mark_as_y()

# Mark features as X
X.skb.mark_as_X()

# Apply TableVectorizer with appropriate encoders per column group
# - numeric columns: use default numeric encoder (handles imputation + scaling)
# - low-cardinality categorical: use OneHotEncoder
# - high-cardinality categorical (Neighborhood): use TargetEncoder
preprocessor = skrub.TableVectorizer(
    numeric_dtype="numeric",
    high_cardinality=skrub.TargetEncoder(min_samples_leaf=20, smoothing=10),
    cardinality_threshold=15  # >15 unique values -> treated as high-cardinality
)

# Apply preprocessor to X
X_processed = X.skb.apply(preprocessor)

# Apply XGBRegressor with exact hyperparameters from original pipeline
pred = X_processed.skb.apply(
    XGBRegressor(
        n_estimators=800,
        max_depth=4,
        learning_rate=0.03,
        random_state=0,
        objective="reg:squarederror",
        missing=np.nan
    ),
    y=y
)

# Export learner
learner = pred.skb.make_learner()