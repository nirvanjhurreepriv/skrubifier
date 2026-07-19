try:
    import stratum as skrub
except ImportError:
    import skrub
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

titanic_df = pd.read_csv("train.csv")
df = skrub.var("titanic", titanic_df)

NUM_COLS = ["Age", "Fare", "SibSp", "Parch"]
CAT_COLS = ["Sex", "Embarked", "Pclass"]
FEATURE_COLS = NUM_COLS + CAT_COLS

X = df[FEATURE_COLS].skb.mark_as_X()
y = df["Survived"].skb.mark_as_y()

vectorized = X.skb.apply(skrub.TableVectorizer())
pred = vectorized.skb.apply(RandomForestClassifier(n_estimators=300, max_depth=8, random_state=0), y=y)

learner = pred.skb.make_learner()
scores = pred.skb.cross_validate(cv=5, scoring="roc_auc")