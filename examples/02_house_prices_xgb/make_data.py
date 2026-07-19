"""
Deterministic synthetic data generator for example 02 (House Prices XGBoost).

Schema matches the Kaggle House Prices competition (subset of columns used
by this pipeline):
  Id, LotArea, OverallQual, YearBuilt, GrLivArea, GarageCars,
  ExterQual, KitchenQual, Neighborhood, SalePrice

Signal: SalePrice is log-normally correlated with OverallQual and GrLivArea
(the strongest predictors in the real dataset).

Usage: python make_data.py  [writes train.csv to this directory]
"""
import os
import numpy as np
import pandas as pd

SEED = 42
N = 1460
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train.csv")

QUAL_CATS = ["Ex", "Gd", "TA", "Fa", "Po"]
NEIGHBORHOODS = [
    "NAmes", "NWAmes", "SawyerW", "StoneBr", "Timber", "Mitchel", "NPkVill",
    "NoRidge", "Crawfor", "Edwards", "SWISU", "Blueste", "MeadowV", "BrDale",
    "ClearCr", "IDOTRR", "OldTown", "Veenker", "CollgCr", "NridgHt",
    "BrkSide", "Gilbert", "Somerst", "Blmngtn", "Sawyer",
]


def make_data(seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # OverallQual: 1-10, roughly normal around 6
    overall_qual = np.clip(rng.normal(6.0, 1.5, N).round().astype(int), 1, 10)

    # YearBuilt: 1872-2009
    year_built = rng.integers(1872, 2010, N)

    # LotArea: log-normal around exp(9) ~ 8100 sq ft
    lot_area = np.clip(rng.lognormal(9.0, 0.5, N).round().astype(int), 1500, 215245)

    # GrLivArea: log-normal around exp(7.1) ~ 1200 sq ft
    gr_liv_area = np.clip(rng.lognormal(7.1, 0.4, N).round().astype(int), 334, 5642)

    # GarageCars: 0-4, mode at 2
    garage_cars = rng.choice([0, 1, 2, 3, 4], N, p=[0.05, 0.20, 0.55, 0.18, 0.02])

    # ExterQual: mostly TA/Gd/Ex distribution matching dataset
    exter_qual_p = np.array([0.37, 0.36, 0.26, 0.01, 0.004])
    exter_qual = rng.choice(QUAL_CATS, N, p=exter_qual_p / exter_qual_p.sum())

    # KitchenQual: similar distribution
    kitchen_p = np.array([0.37, 0.37, 0.24, 0.016, 0.003])
    kitchen_qual = rng.choice(QUAL_CATS, N, p=kitchen_p / kitchen_p.sum())

    # Neighborhood: roughly uniform across 25 neighborhoods
    neighborhood = rng.choice(NEIGHBORHOODS, N)

    # SalePrice: log-normal, strongly correlated with OverallQual and GrLivArea
    # Coefficients calibrated so that mean log_price ~ log(180000) ≈ 12.1
    # Feature contributions at mean values: qual~1.2, gr_liv~2.84, lot~0.45 → sum ~4.64
    # Base constant = 12.1 - 4.64 ≈ 7.46
    log_price = (
        7.5
        + 0.20 * overall_qual
        + 0.40 * np.log(gr_liv_area)
        + 0.05 * np.log(np.maximum(lot_area, 1))
        + 0.10 * (year_built - 1872) / (2009 - 1872)
        + 0.05 * garage_cars
        + rng.normal(0, 0.15, N)  # residual noise
    )
    sale_price = np.clip(np.exp(log_price).round().astype(int), 34900, 755000)

    df = pd.DataFrame({
        "Id": np.arange(1, N + 1),
        "LotArea": lot_area,
        "OverallQual": overall_qual,
        "YearBuilt": year_built,
        "GrLivArea": gr_liv_area,
        "GarageCars": garage_cars.astype(int),
        "ExterQual": exter_qual,
        "KitchenQual": kitchen_qual,
        "Neighborhood": neighborhood,
        "SalePrice": sale_price,
    })
    return df


if __name__ == "__main__":
    df = make_data()
    df.to_csv(OUT, index=False)
    print(f"Wrote {len(df)} rows to {OUT}")
    print(f"  SalePrice range: {df['SalePrice'].min()} - {df['SalePrice'].max()}")
    print(f"  OverallQual range: {df['OverallQual'].min()} - {df['OverallQual'].max()}")
