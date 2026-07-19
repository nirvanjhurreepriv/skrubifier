"""
Deterministic synthetic data generator for example 06 (Allstate Claims Severity).

Schema matches the Kaggle Allstate Claims Severity competition:
  id, cat1..cat116, cont1..cont14, loss

cat columns:  string categories A/B/C/D/E (varying number of levels per column)
cont columns: floats in [0, 1]
loss:         positive float (log-normally distributed)

Signal: loss is partially correlated with a subset of the numeric and
categorical features. The stacking ensemble is expected to recover R² ~0.6.

Usage: python make_data.py  [writes train.csv to this directory]
"""
import os
import numpy as np
import pandas as pd

SEED = 42
N = 2000
N_CAT = 116
N_CONT = 14
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train.csv")


def make_data(seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # Categorical features: each has between 2-5 levels (A/B/..)
    # Some have higher cardinality (up to 5 levels)
    cat_levels = rng.integers(2, 6, N_CAT)  # 2-5 levels per column
    cat_data = {}
    for j in range(N_CAT):
        levels = [chr(65 + k) for k in range(cat_levels[j])]
        # Unequal frequencies (first level most common)
        probs = np.array([3.0 / (k + 3) for k in range(cat_levels[j])])
        probs = probs / probs.sum()
        cat_data[f"cat{j + 1}"] = rng.choice(levels, N, p=probs)

    # Continuous features: uniform [0, 1]
    cont_data = {}
    for j in range(N_CONT):
        cont_data[f"cont{j + 1}"] = rng.uniform(0.0, 1.0, N).round(10)

    # Loss: log-normally distributed; some cont/cat features are predictive
    # Build a latent score from 5 cont features + 3 cat features
    # Continuous signal: cont1 and cont3 are most predictive
    signal = (
        2.0
        + 1.5 * cont_data["cont1"]
        + 1.2 * cont_data["cont3"]
        + 0.8 * cont_data["cont5"]
        + 0.5 * cont_data["cont7"]
        + 0.3 * cont_data["cont9"]
    )
    # Categorical signal: cat1 level 'A' has lower loss
    cat1_effect = np.where(cat_data["cat1"] == "A", -0.3,
                  np.where(cat_data["cat1"] == "B",  0.0, 0.2))
    signal += cat1_effect

    # Add noise and exponentiate to get loss values
    noise = rng.normal(0, 0.6, N)
    loss = np.exp(signal + noise)
    loss = np.clip(loss, 1.0, 200.0).round(2)

    cols = {"id": np.arange(1, N + 1)}
    cols.update(cat_data)
    cols.update(cont_data)
    cols["loss"] = loss

    return pd.DataFrame(cols)


if __name__ == "__main__":
    df = make_data()
    df.to_csv(OUT, index=False)
    print(f"Wrote {len(df)} rows to {OUT}")
    print(f"  loss range: {df['loss'].min():.2f} - {df['loss'].max():.2f}")
    print(f"  cont1 range: {df['cont1'].min():.4f} - {df['cont1'].max():.4f}")
