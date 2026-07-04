try:
    import stratum as skrub
except ImportError:
    import skrub

from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Load the primary table
df = skrub.var("df", None)  # Placeholder; real data provided at runtime

# Extract features and target
X = df.drop(columns=["target"]).skb.mark_as_X()
y = df["target"].skb.mark_as_y()

# Apply transformations: StandardScaler -> SelectKBest -> LogisticRegression
scaled = X.skb.apply(StandardScaler())
selected = scaled.skb.apply(SelectKBest(score_func=f_classif, k=50))
pred = selected.skb.apply(
    LogisticRegression(
        C=0.01,
        class_weight=None,
        dual=False,
        fit_intercept=True,
        intercept_scaling=1,
        l1_ratio=None,
        max_iter=500,
        multi_class="deprecated",
        n_jobs=None,
        penalty="l2",
        random_state=0,
        solver="lbfgs",
        tol=0.0001,
        verbose=0,
        warm_start=False
    ),
    y=y
)

# Export as a learner
learner = pred.skb.make_learner()

# Show CV usage (per original pipeline spec — no CV specified, but adding standard example)
# In practice, replace cv=5 with desired folds and scoring as needed
# scores = pred.skb.cross_validate(cv=5, scoring="roc_auc")