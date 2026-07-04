"""
Converted from examples/08_spooky_author/source_pipeline.py

Key conversion decisions:
- Source uses TfidfVectorizer (max_features=5000, ngram_range=(1,2)) on a
  single text column. The DataOps equivalent uses the same TfidfVectorizer via
  `.skb.apply()` on a text Series, extracted from the input DataFrame with
  `.skb.apply_func(lambda df: df["text"])`. This is the idiomatic DataOps
  pattern for sklearn transformers that expect 1D input (Series) rather than
  a 2D DataFrame.
- TextEncoder would be the most "skrub-native" alternative but requires
  sentence_transformers (not installed). Using the same TfidfVectorizer inside
  DataOps gives a faithful conversion.

Alternative tried — skrub's default text routing (MinHashEncoder):
  `TableVectorizer`'s default high-cardinality routing and
  `skrub.MinHashEncoder(n_components=100)` were both tried as the "pure skrub"
  conversion. On the regenerated overlapping-vocabulary data, MinHashEncoder
  produced original=0.963 vs converted=0.856 (delta 0.107, tolerance 0.029 —
  a hard FAIL). This is a genuine skrub limitation for word-vocabulary text:
  MinHashEncoder hashes character n-grams of the whole text string and produces
  a dense 100-dim vector; it cannot replicate TF-IDF's per-token IDF weighting,
  which is what discriminates between EAP/HPL/MWS on this kind of bag-of-words
  text. The gap does not close with more components — it's structural, not a
  tuning issue. The `TextEncoder` (sentence-transformer embeddings) would likely
  close it, but requires an optional dependency not present in this environment.
- No ColumnTransformer needed: the input DataOp is a single-column text
  DataFrame; the Series extraction step replaces the ColumnTransformer branch.
- 3-class classification (EAP/HPL/MWS) is handled automatically by
  LogisticRegression's multi_class support.

PLAN.md hazard note (CV-loop-internal fitting): the original Kaggle solutions
used leave-one-out target encoding inside a manual CV loop. This simplified
source pipeline (TF-IDF + LR) does NOT have that hazard; it's included in the
test suite to represent the pure-text-input pattern.
"""
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

try:
    import stratum as skrub  # drop-in accelerated backend, same DataOps API
except ImportError:
    import skrub

df_raw = pd.read_csv("train.csv")
spooky = skrub.var("spooky", df_raw)

# Extract text Series from the single-column DataFrame before applying TF-IDF.
# TfidfVectorizer expects 1D input; apply_func routes through the DAG so this
# re-runs correctly at predict time without leaking test-set vocabulary.
X = spooky[["text"]].skb.mark_as_X()
y = spooky["author"].skb.mark_as_y()

text_series = X.skb.apply_func(lambda df: df["text"])
vectorized = text_series.skb.apply(
    TfidfVectorizer(max_features=5000, ngram_range=(1, 2),
                    sublinear_tf=True, stop_words="english")
)
pred = vectorized.skb.apply(
    LogisticRegression(C=5.0, max_iter=1000, random_state=0,
                       multi_class="multinomial", solver="lbfgs"),
    y=y,
)

learner = pred.skb.make_learner()
scores = pred.skb.cross_validate(cv=5, scoring="neg_log_loss")
