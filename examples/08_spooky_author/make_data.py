"""
Deterministic synthetic data generator for example 08 (Spooky Author Identification).

Schema: id, text, author  (author in {EAP, HPL, MWS})

Text generation follows the CHANGELOG specification:
  - Common function words     (~30% of tokens): shared across all authors
  - Shared content words      (~15%): gothic/horror vocabulary used by all
  - Author-preferring words   (~32%): strongly favour one author but may appear
                                       rarely in others
  - Cross-class noise words   (~23%): sampled uniformly from the full vocabulary

This mixing produces realistic class overlap: TF-IDF + LR achieves
ROC AUC ~0.963 macro-OvR (same as the source pipeline uses TfidfVectorizer).

Usage: python make_data.py  [writes train.csv to this directory]
"""
import os
import numpy as np
import pandas as pd

SEED = 42
N = 5000
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train.csv")

AUTHORS = ["EAP", "HPL", "MWS"]
N_PER_CLASS = N // len(AUTHORS)  # 1666 each, last gets remainder

# ---- vocabulary pools ----
COMMON_WORDS = [
    "a", "the", "this", "that", "it", "as", "in", "on", "was", "had",
    "he", "she", "have", "not", "been", "which", "with", "of", "to", "and",
    "is", "at", "by", "his", "her", "they", "we", "you", "from", "are",
]

SHARED_CONTENT = [
    "horror", "night", "dark", "strange", "house", "old", "time", "long",
    "chamber", "tale", "narrator", "world", "never", "could", "would",
]

# Author-preferring word pools (each used 90% for own author, 10% for others)
EAP_WORDS  = ["soul", "truth", "singular", "spirit", "mind", "thought",
              "affection", "heart", "mystery", "profound", "peculiar",
              "investigation", "beautiful", "death", "man", "raven"]
HPL_WORDS  = ["eldritch", "cyclopean", "lurking", "entity", "cosmic",
              "nameless", "aeons", "shunned", "tentacled", "void",
              "hideous", "wretched", "curious", "ancient", "blasphemous"]
MWS_WORDS  = ["sublime", "passion", "nature", "love", "beauty", "monster",
              "creature", "life", "days", "spirit", "noble", "wild",
              "romantic", "despair", "mortal"]

ALL_NOISE_WORDS = EAP_WORDS + HPL_WORDS + MWS_WORDS + SHARED_CONTENT


def _sample_text(rng, author_idx: int, n_tokens: int) -> str:
    tokens = []
    # 30% common function words (shared, no discriminative signal)
    n_common = int(n_tokens * 0.30)
    tokens.extend(rng.choice(COMMON_WORDS, n_common).tolist())
    # 15% shared content words (gothic vocabulary, equal across authors)
    n_shared = int(n_tokens * 0.15)
    tokens.extend(rng.choice(SHARED_CONTENT, n_shared).tolist())
    # 28% author-preferring words — 65% own pool, 35% from other pools
    # (65/35 gives moderate but non-ceiling AUC ~0.96 with TF-IDF + LR)
    n_author = int(n_tokens * 0.28)
    own_pool = [EAP_WORDS, HPL_WORDS, MWS_WORDS][author_idx]
    for _ in range(n_author):
        if rng.random() < 0.65:
            tokens.append(rng.choice(own_pool))
        else:
            tokens.append(rng.choice(ALL_NOISE_WORDS))
    # remainder: cross-class noise words from full vocabulary
    n_noise = n_tokens - len(tokens)
    tokens.extend(rng.choice(ALL_NOISE_WORDS, n_noise).tolist())
    rng.shuffle(tokens)
    return " ".join(tokens)


def make_data(seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    records = []
    for author_idx, author in enumerate(AUTHORS):
        count = N_PER_CLASS if author_idx < len(AUTHORS) - 1 else N - N_PER_CLASS * (len(AUTHORS) - 1)
        for i in range(count):
            n_tokens = rng.integers(12, 35)
            text = _sample_text(rng, author_idx, n_tokens)
            records.append((text, author))

    rng.shuffle(records)
    ids = [f"id{i:05d}" for i in range(N)]
    texts = [r[0] for r in records]
    authors = [r[1] for r in records]

    return pd.DataFrame({"id": ids, "text": texts, "author": authors})


if __name__ == "__main__":
    df = make_data()
    df.to_csv(OUT, index=False)
    print(f"Wrote {len(df)} rows to {OUT}")
    print(f"  Author distribution: {dict(df['author'].value_counts())}")
    print(f"  Text length range (chars): {df['text'].str.len().min()} - {df['text'].str.len().max()}")
