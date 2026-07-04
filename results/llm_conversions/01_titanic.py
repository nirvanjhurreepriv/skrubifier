try:
    import stratum as skrub
except ImportError:
    import skrub
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Load real DataFrame first and pass it to skrub.var()
df = pd.read_csv("train.csv")
table = skrub.var("titanic", df)

# Define feature columns
FEATURE_COLS = ["Age", "Fare", "SibSp", "Parch", "Sex", "Embarked", "Pclass"]

# Mark design matrix and target
X = table[FEATURE_COLS].skb.mark_as_X()
y = table["Survived"].skb.mark_as_y()

# Use TableVectorizer to replace the ColumnTransformer pipeline
# TableVectorizer handles numeric and categorical columns automatically with appropriate encoders
vectorized = X.skb.apply(skrub.TableVectorizer())

# Apply the RandomForestClassifier
pred = vectorized.skb.apply(
    RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        random_state=0
    ),
    y=y
)

# Export as a learner
learner = pred.skb.make_learner()

# Optional: cross-validation (matches typical Kaggle workflow)
scores = pred.skb.cross_validate(cv=5, scoring="roc_auc")