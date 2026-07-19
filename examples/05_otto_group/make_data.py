"""
Deterministic synthetic data generator for example 05 (Otto Group Product Classification).

Schema matches the Kaggle Otto Group competition:
  id, feat_1 ... feat_93, target  (target is 'Class_1' ... 'Class_9')

All 93 features are non-negative integer counts. Each of the 9 classes has a
distinct mean feature profile so a calibrated GBT can learn to discriminate.

Usage: python make_data.py  [writes train.csv to this directory]
"""
import os
import numpy as np
import pandas as pd

SEED = 42
N = 1000
N_CLASSES = 9
N_FEATURES = 93
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train.csv")


def make_data(seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # Balanced class distribution (~111 per class)
    n_per_class = N // N_CLASSES
    classes = []
    for c in range(N_CLASSES):
        count = n_per_class if c < N_CLASSES - 1 else N - n_per_class * (N_CLASSES - 1)
        classes.extend([c] * count)
    classes = np.array(classes)
    rng.shuffle(classes)

    # Each class has a distinct "centroid" in feature space.
    # Features are Poisson counts, mean determined by a class-specific profile.
    # Centroids are designed for moderate (not trivial) class overlap so that a
    # calibrated XGBoost achieves ROC AUC macro ~0.80-0.83 (non-ceiling).
    features = np.zeros((N, N_FEATURES), dtype=int)

    rng_centroids = np.random.default_rng(seed + 1)
    # Base rate for all features (Poisson baseline)
    base_rate = 8.0
    # Class offsets: each class has a slightly different per-feature rate
    # Small offsets (±2) mean moderate overlap between classes
    class_offsets = rng_centroids.uniform(-2.5, 2.5, (N_CLASSES, N_FEATURES))
    # Only the first 45 features carry class-specific signal; the rest are noise
    class_offsets[:, 45:] = 0.0
    centroids = np.clip(base_rate + class_offsets, 0.5, 30.0)

    for i in range(N):
        c = classes[i]
        means = centroids[c]
        features[i] = np.clip(rng.poisson(means), 0, 199)

    target = np.array([f"Class_{c + 1}" for c in classes])

    cols = {"id": np.arange(1, N + 1)}
    for j in range(N_FEATURES):
        cols[f"feat_{j + 1}"] = features[:, j]
    cols["target"] = target

    return pd.DataFrame(cols)


if __name__ == "__main__":
    df = make_data()
    df.to_csv(OUT, index=False)
    print(f"Wrote {len(df)} rows to {OUT}")
    print(f"  Class distribution: {dict(df['target'].value_counts().sort_index())}")
