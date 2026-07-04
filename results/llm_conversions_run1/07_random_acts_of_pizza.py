try:
    import stratum as skrub
except ImportError:
    import skrub
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Load data
df = skrub.var("df", None)  # placeholder; actual data loaded at runtime
X_raw = df.drop(columns=["requester_received_pizza"]).skb.mark_as_X()
y = df["requester_received_pizza"].skb.mark_as_y()

# Extract column groups
title_col = X_raw["request_title"].skb.set_name("title_col")
text_col = X_raw["request_text_edit_aware"].skb.set_name("text_col")
num_cols = X_raw.skb.select(
    skrub.selectors.cols(
        "requester_upvotes_minus_downvotes_at_request",
        "requester_number_of_subreddits_at_request",
        "requester_days_since_first_post_on_reddit",
        "requester_subreddits_at_request_count",
        "number_of_upvotes_of_request_at_retrieval"
    )
).skb.set_name("num_cols")

# Apply TfidfVectorizer to each text column
title_tfidf = title_col.skb.apply(
    TfidfVectorizer(
        analyzer="word",
        binary=False,
        decode_error="strict",
        encoding="utf-8",
        input="content",
        lowercase=True,
        max_df=1.0,
        max_features=500,
        min_df=1,
        norm="l2",
        preprocessor=None,
        smooth_idf=True,
        stop_words="english",
        strip_accents=None,
        sublinear_tf=False,
        token_pattern="(?u)\\b\\w\\w+\\b",
        tokenizer=None,
        use_idf=True,
        vocabulary=None
    )
).skb.set_name("title_tfidf")

text_tfidf = text_col.skb.apply(
    TfidfVectorizer(
        analyzer="word",
        binary=False,
        decode_error="strict",
        encoding="utf-8",
        input="content",
        lowercase=True,
        max_df=1.0,
        max_features=1000,
        min_df=1,
        norm="l2",
        preprocessor=None,
        smooth_idf=True,
        stop_words="english",
        strip_accents=None,
        sublinear_tf=False,
        token_pattern="(?u)\\b\\w\\w+\\b",
        tokenizer=None,
        use_idf=True,
        vocabulary=None
    )
).skb.set_name("text_tfidf")

# Apply StandardScaler to numeric columns
num_scaled = num_cols.skb.apply(
    StandardScaler(copy=True, with_mean=True, with_std=True)
).skb.set_name("num_scaled")

# Combine all transformed features using skrub.TableVectorizer's multi-column support
# Instead of manual ColumnTransformer, use TableVectorizer on the full X with column-specific handling
# However, to preserve the original transformer mapping precisely, we manually combine
# the outputs using sklearn's ColumnTransformer logic translated to skrub:
# Since TableVectorizer can't separately set max_features per column, we replicate the original:
# Create a pipeline that uses the original transformers per column group.
# But per skrub best practices, prefer TableVectorizer if possible. However, the original uses
# max_features=500 for title and max_features=1000 for text — which TableVectorizer cannot do per column.
# Therefore, we keep the original per-column transformers and combine the outputs.
from sklearn.compose import ColumnTransformer

# Create a ColumnTransformer-like structure using skrub.apply on the whole frame
# But skrub doesn't support ColumnTransformer inside apply directly. Instead, use per-column apply +.hstack
# However, skrub's apply supports sklearn ColumnTransformer if we pass it as an estimator.
# But the IR specifies apply to individual column groups.
# Therefore, the correct way per the rules is to use TableVectorizer with explicit col types — but the IR requires separate max_features.
# Since the IR has explicit ColumnTransformer logic and skrub supports ColumnTransformer as a transformer:
pre = ColumnTransformer([
    ("title_tfidf", TfidfVectorizer(
        analyzer="word",
        binary=False,
        decode_error="strict",
        encoding="utf-8",
        input="content",
        lowercase=True,
        max_df=1.0,
        max_features=500,
        min_df=1,
        norm="l2",
        preprocessor=None,
        smooth_idf=True,
        stop_words="english",
        strip_accents=None,
        sublinear_tf=False,
        token_pattern="(?u)\\b\\w\\w+\\b",
        tokenizer=None,
        use_idf=True,
        vocabulary=None
    ), "request_title"),
    ("text_tfidf", TfidfVectorizer(
        analyzer="word",
        binary=False,
        decode_error="strict",
        encoding="utf-8",
        input="content",
        lowercase=True,
        max_df=1.0,
        max_features=1000,
        min_df=1,
        norm="l2",
        preprocessor=None,
        smooth_idf=True,
        stop_words="english",
        strip_accents=None,
        sublinear_tf=False,
        token_pattern="(?u)\\b\\w\\w+\\b",
        tokenizer=None,
        use_idf=True,
        vocabulary=None
    ), "request_text_edit_aware"),
    ("num", StandardScaler(copy=True, with_mean=True, with_std=True), [
        "requester_upvotes_minus_downvotes_at_request",
        "requester_number_of_subreddits_at_request",
        "requester_days_since_first_post_on_reddit",
        "requester_subreddits_at_request_count",
        "number_of_upvotes_of_request_at_retrieval"
    ])
])

vectorized = X_raw.skb.apply(pre).skb.set_name("vectorized")

# Apply the final estimator
pred = vectorized.skb.apply(
    LogisticRegression(
        C=0.1,
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
).skb.set_name("prediction")

# Export the learner
learner = pred.skb.make_learner()