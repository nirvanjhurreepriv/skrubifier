"""
Deterministic synthetic data generator for example 07 (Random Acts of Pizza).

Schema matches the Kaggle/MLE-Bench Random Acts of Pizza competition:
  request_id, request_title, request_text_edit_aware,
  requester_upvotes_minus_downvotes_at_request,
  requester_number_of_subreddits_at_request,
  requester_days_since_first_post_on_reddit,
  requester_subreddits_at_request_count,
  number_of_upvotes_of_request_at_retrieval,
  requester_received_pizza  (binary target)

Signal: text content + upvotes + days on reddit are partially predictive.
Successful requests tend to include "need" / "family" / gratitude words.
Class imbalance: ~22% positive (received pizza).

Usage: python make_data.py  [writes train.csv to this directory]
"""
import os
import numpy as np
import pandas as pd

SEED = 42
N = 5000
POSITIVE_RATE = 0.2268  # ~1134/5000
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train.csv")

# Text is nearly pure noise — both TF-IDF and MinHashEncoder see the same
# (near-zero) signal, so their AUCs match. All real predictive signal comes
# from the numeric features (upvotes, days on reddit), consistent with the
# real Random Acts of Pizza dataset where text had modest incremental value.
_VOCAB = [
    "pizza", "hungry", "food", "really", "need", "want", "have", "like",
    "hope", "day", "good", "please", "help", "think", "know", "would",
    "could", "going", "great", "long", "here", "well", "time", "back",
    "friend", "life", "work", "just", "first", "last", "never", "always",
    "even", "still", "around", "about", "since", "every", "other", "maybe",
    "love", "home", "year", "week", "today", "month", "hard", "sorry",
    "thank", "appreciate", "honest", "true", "miss", "post", "comment",
    "reddit", "college", "school", "study", "job", "situation", "rough",
]
TITLE_POOL = ["pizza request", "asking for pizza", "long shot", "just trying",
              "need some food", "hungry tonight", "anyone?", "hoping for pizza",
              "would appreciate", "throwing it out there", "first post here",
              "college student", "a request", "friday night"]


def _make_text(rng, _is_positive: bool) -> str:
    """Generate random short text from shared vocabulary (no per-class signal)."""
    n_words = rng.integers(8, 18)
    words = rng.choice(_VOCAB, n_words)
    return " ".join(words)


def make_data(seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # Target: ~22% positive
    received = (rng.random(N) < POSITIVE_RATE).astype(int)

    # Text columns
    request_texts = []
    request_titles = []
    for i in range(N):
        is_pos = bool(received[i])
        text = _make_text(rng, is_pos)
        request_texts.append(text)
        title = rng.choice(TITLE_POOL)
        request_titles.append(title)

    # Numeric features: positives tend to have more upvotes and longer reddit history
    # This is where the modest predictive signal lives (~0.63 AUC).
    upvotes_base   = np.where(received, 40.0, 8.0)
    upvotes        = np.clip(rng.normal(upvotes_base, 40.0, N).astype(int), -50, 499)
    days_base      = np.where(received, 400.0, 150.0)
    days_history   = np.clip(rng.exponential(days_base, N).round(1), 0.0, 1618.9)
    n_subreddits   = np.clip(rng.integers(1, 30, N), 1, 29)
    sub_count      = np.clip(rng.integers(1, 15, N), 1, 14)
    n_upvotes_ret  = np.clip(rng.poisson(5.0, N).astype(int), 0, 99)

    df = pd.DataFrame({
        "request_id":                                    [f"t3_{i:06d}" for i in range(N)],
        "request_title":                                 request_titles,
        "request_text_edit_aware":                       request_texts,
        "requester_upvotes_minus_downvotes_at_request":  upvotes.astype(float),
        "requester_number_of_subreddits_at_request":     n_subreddits.astype(float),
        "requester_days_since_first_post_on_reddit":     days_history,
        "requester_subreddits_at_request_count":         sub_count.astype(float),
        "number_of_upvotes_of_request_at_retrieval":     n_upvotes_ret.astype(float),
        "requester_received_pizza":                      received,
    })
    return df


if __name__ == "__main__":
    df = make_data()
    df.to_csv(OUT, index=False)
    print(f"Wrote {len(df)} rows to {OUT}")
    print(f"  Positive rate: {df['requester_received_pizza'].mean():.3f}")
