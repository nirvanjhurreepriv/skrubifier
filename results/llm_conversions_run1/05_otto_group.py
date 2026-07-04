try:
    import stratum as skrub
except ImportError:
    import skrub

from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV

# Declare the primary input table
df = skrub.var("df", None)

# Extract features and target
FEATURE_COLUMNS = [f"feat_{i}" for i in range(1, 94)]
TARGET_COLUMN = "target"

X = df.skb.select(skrub.selectors.cols(*FEATURE_COLUMNS))
y = df[TARGET_COLUMN]

# Mark as X and y for the learner
X.skb.mark_as_X()
y.skb.mark_as_y()

# Encode target labels (0..8) as required by XGBoost multi:softprob objective
y_encoded = y.skb.apply_func(LabelEncoder().fit_transform)

# Create the estimator with calibrated classifier
estimator = CalibratedClassifierCV(
    XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        num_class=9,
        random_state=0,
    ),
    method="isotonic",
    cv=10
)

# Apply estimator with encoded target
pred = X.skb.apply(estimator, y=y_encoded)

# Export as learner
learner = pred.skb.make_learner()