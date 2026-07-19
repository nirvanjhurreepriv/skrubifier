try:
    import stratum as skrub
except ImportError:
    import skrub

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Load the data and declare it as a pipeline input
df_raw = pd.read_csv("train.csv")
df = skrub.var("df", df_raw)

# Feature / target selection
FEATURE_COLUMNS = ["Age", "Fare", "SibSp", "Parch", "Sex", "Embarked", "Pclass"]
X = df[FEATURE_COLUMNS].skb.mark_as_X()
y = df["Survived"].skb.mark_as_y()

# Automatic preprocessing: impute, encode, scale
preprocessed = X.skb.apply(skrub.TableVectorizer())

# Model
pred = preprocessed.skb.apply(
    RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        random_state=0,
        bootstrap=True,
        ccp_alpha=0.0,
        class_weight=None,
        criterion="gini",
        max_features="sqrt",
        max_leaf_nodes=None,
        max_samples=None,
        min_impurity_decrease=0.0,
        min_samples_leaf=1,
        min_samples_split=2,
        min_weight_fraction_leaf=0.0,
        monotonic_cst=None,
        n_jobs=None,
        oob_score=False,
        verbose=0,
        warm_start=False,
    ),
    y=y,
)

# Export the DAG as a learner
learner = pred.skb.make_learner()