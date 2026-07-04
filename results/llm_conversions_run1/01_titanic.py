try:
    import stratum as skrub
except ImportError:
    import skrub
from sklearn.ensemble import RandomForestClassifier

# Declare the primary table
df = skrub.var("df", None)  # Placeholder; actual data loaded at runtime

# Mark X and y
target_col = "Survived"
X = df.drop(columns=[target_col]).skb.mark_as_X()
y = df[target_col].skb.mark_as_y()

# Apply TableVectorizer which subsumes the column-specific preprocessing
vectorized = X.skb.apply(skrub.TableVectorizer())

# Apply the classifier
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

# Export the learner
learner = pred.skb.make_learner()

# CV evaluation (since the original code didn't specify but is standard practice)
scores = pred.skb.cross_validate(cv=5, scoring="roc_auc")