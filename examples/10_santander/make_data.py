"""
Deterministic synthetic data generator for example 10 (Santander Customer
Transaction Prediction).

Schema: ID_code, var_0 ... var_199, target  (target is binary, ~8% positive)

All 200 features are real-valued. The real competition's defining property:
each feature is approximately independent across target classes (no feature
interactions), so simple SelectKBest + LogisticRegression works well.

Signal: ~50 of the 200 features are weakly correlated with the target.
The remaining 150 are pure noise. SelectKBest(f_classif, k=50) recovers
the signal features.

Usage: python make_data.py  [writes train.csv to this directory]
"""
import os
import numpy as np
import pandas as pd

SEED = 42
N = 5000
N_FEATURES = 200
N_SIGNAL = 50       # first 50 features carry signal
POSITIVE_RATE = 0.0758  # ~379/5000 matching original data
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train.csv")


def make_data(seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # Target: ~8% positive
    target = (rng.random(N) < POSITIVE_RATE).astype(int)

    # Features: N(0, 1) base, with positive class having slightly shifted mean
    # for the first N_SIGNAL features
    features = rng.standard_normal((N, N_FEATURES))

    # Signal features: positive class has a very small mean shift of +0.05
    # (mimics the real Santander data where marginal per-feature shifts are tiny).
    pos_mask = target == 1
    features[np.ix_(pos_mask, np.arange(N_SIGNAL))] += 0.05

    cols = {"ID_code": [f"test_{i:06d}" for i in range(N)]}
    for j in range(N_FEATURES):
        cols[f"var_{j}"] = features[:, j].round(10)
    cols["target"] = target

    return pd.DataFrame(cols)


if __name__ == "__main__":
    df = make_data()
    df.to_csv(OUT, index=False)
    print(f"Wrote {len(df)} rows to {OUT}")
    print(f"  Positive rate: {df['target'].mean():.4f}")
    print(f"  var_0 range: {df['var_0'].min():.4f} - {df['var_0'].max():.4f}")
