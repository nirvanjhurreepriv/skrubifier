"""
Converted from examples/07_random_acts_of_pizza/source_pipeline.py

Key conversion decisions:
- The source pipeline uses TfidfVectorizer on two text columns plus StandardScaler
  on numeric columns via ColumnTransformer.
- skrub.TableVectorizer auto-detects column types: string/high-cardinality
  columns get MinHashEncoder (or similar), numeric get scale/impute treatment.
  This subsumes the entire ColumnTransformer in one call, which is the intended
  DataOps pattern.
- skrub.TextEncoder (sentence-transformer embeddings) would be ideal for long
  text but requires the optional `sentence_transformers` package; TableVectorizer
  falls back to MinHashEncoder for string columns which is always available and
  provides a good baseline for bag-of-words-level text features.
- Note: dtype_hint="text" is recorded in the IR for the two text columns;
  the LLM converter should prefer MinHashEncoder/StringEncoder over
  TableVectorizer's default when the columns are flagged as text.
"""
import pandas as pd

try:
    import stratum as skrub  # drop-in accelerated backend, same DataOps API
except ImportError:
    import skrub

from sklearn.linear_model import LogisticRegression

TEXT_COLS = ["request_title", "request_text_edit_aware"]
NUM_COLS = [
    "requester_upvotes_minus_downvotes_at_request",
    "requester_number_of_subreddits_at_request",
    "requester_days_since_first_post_on_reddit",
    "requester_subreddits_at_request_count",
    "number_of_upvotes_of_request_at_retrieval",
]
FEATURE_COLUMNS = TEXT_COLS + NUM_COLS

df_raw = pd.read_csv("train.csv")
pizza = skrub.var("pizza", df_raw)

X = pizza[FEATURE_COLUMNS].skb.mark_as_X()
y = pizza["requester_received_pizza"].skb.mark_as_y()

# TableVectorizer handles both text (string) and numeric columns automatically:
# string cols -> MinHashEncoder (n_components=30 by default in skrub 0.9);
# numeric cols -> impute + passthrough.
vectorized = X.skb.apply(skrub.TableVectorizer())
pred = vectorized.skb.apply(
    LogisticRegression(C=0.1, max_iter=500, random_state=0), y=y
)

learner = pred.skb.make_learner()
scores = pred.skb.cross_validate(cv=5, scoring="roc_auc")
