"""
Adapted from the Kaggle "Random Acts of Pizza" competition (MLE-Bench).
Binary classification: predict whether a reddit user received a pizza donation.

Features: two free-text columns (request_title, request_text_edit_aware)
alongside tabular numeric features (upvotes, days since first post, etc.).
This is the key diversity axis: it requires TextEncoder (or similar) for the
text columns and TableVectorizer for the numeric ones, testing the
IR's dtype_hint="text" path and multi-transformer branch handling.

The source pipeline uses TfidfVectorizer on concatenated text + numeric
features via ColumnTransformer.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

TEXT_COLS = ["request_title", "request_text_edit_aware"]
NUM_COLS = [
    "requester_upvotes_minus_downvotes_at_request",
    "requester_number_of_subreddits_at_request",
    "requester_days_since_first_post_on_reddit",
    "requester_subreddits_at_request_count",
    "number_of_upvotes_of_request_at_retrieval",
]
FEATURE_COLUMNS = TEXT_COLS + NUM_COLS
TARGET_COLUMN = "requester_received_pizza"
TASK = "classification"

preprocessor = ColumnTransformer([
    ("title_tfidf", TfidfVectorizer(max_features=500, stop_words="english"), "request_title"),
    ("text_tfidf", TfidfVectorizer(max_features=1000, stop_words="english"),
     "request_text_edit_aware"),
    ("num", StandardScaler(), NUM_COLS),
])

PIPELINE = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LogisticRegression(C=0.1, max_iter=500, random_state=0)),
])

if __name__ == "__main__":
    df = pd.read_csv("train.csv")
    X, y = df[FEATURE_COLUMNS], df[TARGET_COLUMN]
    PIPELINE.fit(X, y)
