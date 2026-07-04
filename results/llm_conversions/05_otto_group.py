import skrub
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV

def load_data():
    # Load a small subset for demonstration
    n_samples = 100
    np.random.seed(0)
    df = pd.DataFrame({
        f"feat_{i}": np.random.randn(n_samples) for i in range(1, 94)
    })
    df["target"] = np.random.choice([f"Class_{i}" for i in range(1, 10)], size=n_samples)
    return df

# Load data lazily using skrub.var
data = skrub.var("data", load_data)

CLASSES = [f"Class_{i}" for i in range(1, 10)]

le = LabelEncoder().fit(CLASSES)

def preprocess(data):
    data = data.copy()
    data["target_encoded"] = le.transform(data["target"])
    return data

data = data.skb.apply(preprocess)

FEATURE_COLUMNS = [f"feat_{i}" for i in range(1, 94)]

X = data[FEATURE_COLUMNS].skb.mark_as_X()
y = data["target_encoded"].skb.mark_as_y()

base_estimator = XGBClassifier(
    n_estimators=50,
    max_depth=3,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="multi:softprob",
    num_class=9,
    random_state=0,
    n_jobs=-1
)

model = CalibratedClassifierCV(base_estimator, method="isotonic", cv=3)

pred = X.skb.apply(model, y=y, predict_proba=False)

learner = pred.skb.make_learner()