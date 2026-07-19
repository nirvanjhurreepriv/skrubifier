"""
Deterministic synthetic data generator for example 01 (Titanic-style survival).

Schema matches the real Kaggle Titanic training set:
  PassengerId, Survived, Pclass, Sex, Age (nullable), SibSp, Parch, Fare, Embarked (nullable)

Signal: survival probability follows the well-known Titanic pattern
  (sex, passenger class, and age are the main predictors).

Usage: python make_data.py  [writes train.csv to this directory]
"""
import os
import numpy as np
import pandas as pd

SEED = 42
N = 891
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train.csv")


def make_data(seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # Passenger class: roughly 27% 1st, 21% 2nd, 52% 3rd
    pclass = rng.choice([1, 2, 3], size=N, p=[0.268, 0.205, 0.527])

    # Sex: 35% female
    is_female = rng.random(N) < 0.354
    sex = np.where(is_female, "female", "male")

    # Age: roughly uniform 2-70, ~17% missing
    age = rng.uniform(2.0, 70.0, N).round(2)
    age_na = rng.random(N) < (154 / 891)
    age = age.astype(float)
    age[age_na] = np.nan

    # SibSp and Parch (family sizes)
    sibsp = rng.choice([0, 1, 2, 3, 4], size=N, p=[0.68, 0.23, 0.055, 0.025, 0.01])
    parch = rng.choice([0, 1, 2, 3], size=N, p=[0.77, 0.13, 0.07, 0.03])

    # Fare: roughly correlated with class (higher class -> higher fare)
    fare_base = np.where(pclass == 1, 80.0, np.where(pclass == 2, 20.0, 10.0))
    fare = np.clip(rng.lognormal(np.log(fare_base), 0.7, N), 5.0, 300.0).round(4)

    # Embarkation port: S=70%, C=20%, Q=8%, NaN=2%
    embarked_idx = rng.choice([0, 1, 2, 3], size=N, p=[0.70, 0.20, 0.08, 0.02])
    embarked_map = {0: "S", 1: "C", 2: "Q", 3: None}
    embarked = np.array([embarked_map[i] for i in embarked_idx], dtype=object)

    # Survival: logistic model matching known Titanic patterns
    log_odds = (
        -0.4
        + is_female.astype(float) * 2.4     # women survive much more
        - (pclass == 3).astype(float) * 1.2 # 3rd class: big disadvantage
        + (pclass == 1).astype(float) * 0.8 # 1st class: advantage
    )
    # Add age effect: children survive more, older adults slightly less
    age_filled = np.where(age_na, 30.0, age)
    log_odds += np.where(age_filled < 10, 0.8, 0.0)
    log_odds -= 0.01 * np.maximum(age_filled - 30, 0)

    p_survive = 1.0 / (1.0 + np.exp(-log_odds))
    survived = rng.binomial(1, p_survive).astype(int)

    df = pd.DataFrame({
        "PassengerId": np.arange(1, N + 1),
        "Survived": survived,
        "Pclass": pclass.astype(int),
        "Sex": sex,
        "Age": age,
        "SibSp": sibsp.astype(int),
        "Parch": parch.astype(int),
        "Fare": fare,
        "Embarked": embarked,
    })
    return df


if __name__ == "__main__":
    df = make_data()
    df.to_csv(OUT, index=False)
    print(f"Wrote {len(df)} rows to {OUT}")
    print(f"  Survived rate: {df['Survived'].mean():.3f}")
    print(f"  Age NaN: {df['Age'].isna().sum()}")
