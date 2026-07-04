try:
    import stratum as skrub
except ImportError:
    import skrub
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load real DataFrame before creating skrub.var
df = pd.read_csv("train.csv")

# Declare pipeline input - pass real DataFrame as second argument
df_op = skrub.var("df", df)

# Extract features and target - always assign mark_as_X and mark_as_y return values
X = df_op["text"].skb.mark_as_X()
y = df_op["author"].skb.mark_as_y()

# Apply TF-IDF vectorization via .skb.apply() (stateful transformer with fit/transform)
vectorized = X.skb.apply(TfidfVectorizer(
    analyzer="word",
    binary=False,
    decode_error="strict",
    encoding="utf-8",
    input="content",
    lowercase=True,
    max_df=1.0,
    max_features=5000,
    min_df=1,
    norm="l2",
    preprocessor=None,
    smooth_idf=True,
    stop_words="english",
    strip_accents=None,
    sublinear_tf=True,
    token_pattern="(?u)\\b\\w\\w+\\b",
    tokenizer=None,
    use_idf=True,
    vocabulary=None
))

# Apply LogisticRegression classifier - pass y for proper labeling
pred = vectorized.skb.apply(LogisticRegression(
    C=5.0,
    class_weight=None,
    dual=False,
    fit_intercept=True,
    intercept_scaling=1,
    l1_ratio=None,
    max_iter=1000,
    multi_class="multinomial",
    n_jobs=None,
    penalty="l2",
    random_state=0,
    solver="lbfgs",
    tol=0.0001,
    verbose=0,
    warm_start=False
), y=y)

# Export as SkrubLearner
learner = pred.skb.make_learner()

# Optional: add cross-validation for evaluation (as commonly done in Kaggle pipelines)
scores = pred.skb.cross_validate(cv=5, scoring="accuracy")