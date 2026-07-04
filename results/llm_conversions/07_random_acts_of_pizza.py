try:
    import stratum as skrub
except ImportError:
    import skrub
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Load real dataframes and create DataOps
df = pd.read_csv("train.csv")
data = skrub.var("df", df)

# Define column groups per the IR
text_title_cols = ["request_title"]
text_text_cols = ["request_text_edit_aware"]
num_cols = [
    "requester_upvotes_minus_downvotes_at_request",
    "requester_number_of_subreddits_at_request",
    "requester_days_since_first_post_on_reddit",
    "requester_subreddits_at_request_count",
    "number_of_upvotes_of_request_at_retrieval",
]

# Helper to convert TfidfVectorizer output to DataFrame
def make_tfidf_dataframe_output(transformer):
    class TfidfDataFrame(TfidfVectorizer):
        def fit_transform(self, X, y=None, **fit_params):
            # Ensure X is a pandas Series and extract values
            if isinstance(X, pd.Series):
                X_values = X.values
            elif isinstance(X, pd.DataFrame):
                X_values = X.iloc[:, 0].values
            else:
                X_values = X
            
            result = super().fit_transform(X_values, y, **fit_params)
            
            if isinstance(X, pd.Series):
                feature_names = [f"{X.name}_{i}" for i in range(result.shape[1])]
                return pd.DataFrame(result.toarray(), columns=feature_names, index=X.index)
            elif isinstance(X, pd.DataFrame):
                feature_names = [f"{X.columns[0]}_{i}" for i in range(result.shape[1])]
                return pd.DataFrame(result.toarray(), columns=feature_names, index=X.index)
            else:
                return result

        def transform(self, X):
            # Ensure X is a pandas Series and extract values
            if isinstance(X, pd.Series):
                X_values = X.values
            elif isinstance(X, pd.DataFrame):
                X_values = X.iloc[:, 0].values
            else:
                X_values = X
            
            result = super().transform(X_values)
            
            if isinstance(X, pd.Series):
                feature_names = [f"{X.name}_{i}" for i in range(result.shape[1])]
                return pd.DataFrame(result.toarray(), columns=feature_names, index=X.index)
            elif isinstance(X, pd.DataFrame):
                feature_names = [f"{X.columns[0]}_{i}" for i in range(result.shape[1])]
                return pd.DataFrame(result.toarray(), columns=feature_names, index=X.index)
            else:
                return result

        def get_feature_names_out(self, input_features=None):
            col_name = input_features[0] if input_features and isinstance(input_features, (list, tuple)) else "text"
            return [f"{col_name}_{i}" for i in range(self.idf_.shape[0])]

    return TfidfDataFrame(**transformer.get_params())

# Process each text column separately with its own TF-IDF vectorizer
title_vectorized = data[text_title_cols].skb.apply(
    make_tfidf_dataframe_output(
        TfidfVectorizer(
            max_features=500,
            stop_words="english",
            analyzer="word",
            binary=False,
            decode_error="strict",
            encoding="utf-8",
            input="content",
            lowercase=True,
            max_df=1.0,
            min_df=1,
            norm="l2",
            preprocessor=None,
            smooth_idf=True,
            strip_accents=None,
            sublinear_tf=False,
            token_pattern="(?u)\\b\\w\\w+\\b",
            tokenizer=None,
            use_idf=True,
            vocabulary=None
        )
    ),
    y=None
)

text_vectorized = data[text_text_cols].skb.apply(
    make_tfidf_dataframe_output(
        TfidfVectorizer(
            max_features=1000,
            stop_words="english",
            analyzer="word",
            binary=False,
            decode_error="strict",
            encoding="utf-8",
            input="content",
            lowercase=True,
            max_df=1.0,
            min_df=1,
            norm="l2",
            preprocessor=None,
            smooth_idf=True,
            strip_accents=None,
            sublinear_tf=False,
            token_pattern="(?u)\\b\\w\\w+\\b",
            tokenizer=None,
            use_idf=True,
            vocabulary=None
        )
    ),
    y=None
)

# Process numeric columns
numeric = data[num_cols].skb.apply(StandardScaler(copy=True, with_mean=True, with_std=True))

# Concatenate all processed features horizontally
X = skrub.concat([title_vectorized, text_vectorized, numeric], axis=1).skb.mark_as_X()

# Extract target
y = data["requester_received_pizza"].skb.mark_as_y()

# Apply final estimator
pred = X.skb.apply(
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
)

# Export as learner and run cross-validation
learner = pred.skb.make_learner()
scores = pred.skb.cross_validate(cv=5, scoring="roc_auc")