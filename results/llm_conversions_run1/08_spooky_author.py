try:
    import stratum as skrub
except ImportError:
    import skrub
import pandas as pd
from sklearn.linear_model import LogisticRegression

# Load data
df = skrub.var("df", pd.read_csv("train.csv"))

# Mark X and y
X = df["text"].to_frame().skb.mark_as_X()
y = df["author"].skb.mark_as_y()

# Apply TF-IDF using TableVectorizer with TextEncoder (equivalent to TfidfVectorizer)
# TfidfVectorizer params: analyzer='word', binary=False, decode_error='strict', 
# encoding='utf-8', input='content', lowercase=True, max_df=1.0, max_features=5000,
# min_df=1, norm='l2', preprocessor=None, smooth_idf=True, stop_words='english',
# strip_accents=None, sublinear_tf=True, token_pattern='(?u)\b\w\w+\b', tokenizer=None,
# use_idf=True, vocabulary=None
vectorized = X.skb.apply(skrub.TableVectorizer(
    text_transformer=skrub.TextEncoder(
        encoding="utf-8",
        lowercase=True,
        max_features=5000,
        min_df=1,
        norm="l2",
        smooth_idf=True,
        stop_words="english",
        sublinear_tf=True,
        use_idf=True,
    )
))

# Apply logistic regression
pred = vectorized.skb.apply(
    LogisticRegression(
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
        warm_start=False,
    ),
    y=y
)

# Export learner
learner = pred.skb.make_learner()

# CV loop (as per original Kaggle evaluation practice)
scores = pred.skb.cross_validate(cv=5, scoring="accuracy")