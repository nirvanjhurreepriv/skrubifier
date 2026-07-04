"""
Adapted from Kaggle "Spooky Author Identification" competition.
3-class text classification: predict the author (EAP / HPL / MWS) of a
gothic-horror sentence.

This example tests the CV-loop-internal fitting hazard from PLAN.md §3:
the original Kaggle solutions often used leave-one-out or k-fold target
encoding INSIDE the CV loop. Here we simplify to a clean TF-IDF + LR
pipeline (no leakage risk) but note the full pattern in the docstring.

The key conversion challenge: a pure-text single-column input pipeline where
skrub's TextEncoder (or MinHashEncoder) replaces TfidfVectorizer. The
ColumnTransformer has only one branch (the text column), which is an edge
case the analyzer must handle (no "column group" diversity).
"""
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

TEXT_COL = "text"
FEATURE_COLUMNS = [TEXT_COL]
TARGET_COLUMN = "author"
TASK = "classification"

PIPELINE = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2),
                              sublinear_tf=True, stop_words="english")),
    ("model", LogisticRegression(C=5.0, max_iter=1000, random_state=0,
                                  multi_class="multinomial", solver="lbfgs")),
])

if __name__ == "__main__":
    df = pd.read_csv("train.csv")
    X, y = df[TEXT_COL], df[TARGET_COLUMN]
    PIPELINE.fit(X, y)
