try:
    import stratum as skrub
except ImportError:
    import skrub
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Load real DataFrame; pass it as the var default (required)
df = pd.read_csv("train.csv")
table = skrub.var("df", df)

# Extract feature columns and target as specified
FEATURE_COLUMNS = [f"var_{i}" for i in range(200)]
TARGET_COLUMN = "target"

# Mark X and y
X = table[FEATURE_COLUMNS].skb.mark_as_X()
y = table[TARGET_COLUMN].skb.mark_as_y()

# Apply StandardScaler (stateful transformer via .skb.apply)
scaled = X.skb.apply(StandardScaler())

# Apply SelectKBest (stateful transformer via .skb.apply) with y
selected = scaled.skb.apply(SelectKBest(score_func=f_classif, k=50), y=y)

# Apply LogisticRegression (stateful estimator via .skb.apply with y)
pred = selected.skb.apply(LogisticRegression(C=0.01, max_iter=500, random_state=0), y=y)

# Export the DAG as a SkrubLearner
learner = pred.skb.make_learner()