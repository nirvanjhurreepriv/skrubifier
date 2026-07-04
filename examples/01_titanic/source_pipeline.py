"""
Representative of the most common MLE-Bench / Kaggle tabular pattern:
ColumnTransformer(numeric + categorical branches) -> RandomForest, single table.
(Pattern matches e.g. the widely-copied Titanic baseline referenced in the
Kaggle "winning solutions" notebook collection.)
"""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUM_COLS = ["Age", "Fare", "SibSp", "Parch"]
CAT_COLS = ["Sex", "Embarked", "Pclass"]

TASK = "classification"
FEATURE_COLUMNS = NUM_COLS + CAT_COLS
TARGET_COLUMN = "Survived"

preprocessor = ColumnTransformer([
    ("num", Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ]), NUM_COLS),
    ("cat", Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore")),
    ]), CAT_COLS),
])

PIPELINE = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(n_estimators=300, max_depth=8, random_state=0)),
])

if __name__ == "__main__":
    df = pd.read_csv("train.csv")
    X, y = df[FEATURE_COLUMNS], df[TARGET_COLUMN]
    PIPELINE.fit(X, y)
