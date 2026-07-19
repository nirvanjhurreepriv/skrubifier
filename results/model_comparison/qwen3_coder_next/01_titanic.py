try:
    import stratum as skrub
except ImportError:
    import skrub
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Load real data first (required for skrub.var)
df = pd.read_csv("train.csv")

# Declare the primary input DataOp with real DataFrame
df_dataop = skrub.var("df", df)

# Define column groups as per IR
NUM_COLS = ["Age", "Fare", "SibSp", "Parch"]
CAT_COLS = ["Sex", "Embarked", "Pclass"]
FEATURE_COLS = NUM_COLS + CAT_COLS

# Mark features and target
X = df_dataop[FEATURE_COLS].skb.mark_as_X()
y = df_dataop["Survived"].skb.mark_as_y()

# Use TableVectorizer to replace ColumnTransformer with separate pipelines
# (TableVectorizer subsumes both numeric and categorical preprocessing)
vectorized = X.skb.apply(skrub.TableVectorizer())

# Apply the estimator with y parameter for supervised learning
pred = vectorized.skb.apply(
    RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        random_state=0,
        bootstrap=True,
        criterion="gini",
        max_features="sqrt",
        min_samples_leaf=1,
        min_samples_split=2
    ),
    y=y
)

# Export as learner and run cross-validation
learner = pred.skb.make_learner()
scores = pred.skb.cross_validate(cv=5, scoring="roc_auc")