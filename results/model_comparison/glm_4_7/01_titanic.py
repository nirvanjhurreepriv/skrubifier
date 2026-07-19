try:
    import stratum as skrub
except ImportError:
    import skrub

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Load the real DataFrame eagerly
df = pd.read_csv("train.csv")

# Define the variable using the loaded DataFrame (required for schema inference)
# Variable name 'df' matches the IR role "primary"
var_df = skrub.var("df", df)

# Define feature columns based on the IR's column groups and the original source
NUM_COLS = ["Age", "Fare", "SibSp", "Parch"]
CAT_COLS = ["Sex", "Embarked", "Pclass"]
FEATURE_COLS = NUM_COLS + CAT_COLS

# Select feature columns from the DataOp
X = var_df[FEATURE_COLS].skb.mark_as_X()

# Define target column exactly as specified
y = var_df["Survived"].skb.mark_as_y()

# Apply the skrub TableVectorizer. 
# This subsumes the SimpleImputer + StandardScaler for numeric 
# and SimpleImputer + OneHotEncoder for categorical columns.
vectorized = X.skb.apply(skrub.TableVectorizer())

# Apply the RandomForestClassifier with hyperparameters from the IR
pred = vectorized.skb.apply(
    RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        random_state=0
    ),
    y=y
)

# Make the learner
learner = pred.skb.make_learner()