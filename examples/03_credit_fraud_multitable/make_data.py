"""
Deterministic synthetic data generator for example 03 (Credit Fraud multi-table).

Two tables:
  baskets.csv  — one row per basket: basket_ID, fraud_flag, customer_age
  products.csv — 4 rows per basket: basket_ID, cash_price, product_category

Signal: fraudulent baskets have systematically higher product prices
(both mean and max are elevated), which the HistGBT classifier can learn
via the aggregated features (price_mean, price_max, price_sum, n_items).

Usage: python make_data.py  [writes baskets.csv and products.csv to this directory]
"""
import os
import numpy as np
import pandas as pd

SEED = 42
N_BASKETS = 600
PRODUCTS_PER_BASKET = 4  # exactly 4 → 2400 products total
FRAUD_RATE = 0.30
CATEGORIES = ["electronics", "clothing", "food", "toys", "sports"]
OUT_DIR = os.path.dirname(os.path.abspath(__file__))


def make_data(seed: int = SEED):
    rng = np.random.default_rng(seed)

    # --- baskets ---
    basket_ids = np.arange(1, N_BASKETS + 1)
    fraud_flag = (rng.random(N_BASKETS) < FRAUD_RATE).astype(int)
    customer_age = rng.integers(18, 75, N_BASKETS)

    baskets_df = pd.DataFrame({
        "basket_ID": basket_ids,
        "fraud_flag": fraud_flag,
        "customer_age": customer_age,
    })

    # --- products ---
    # Non-fraud: prices log-normal around exp(3) ~ 20, sigma 0.6
    # Fraud:     prices log-normal around exp(4) ~ 55, sigma 0.8  (higher prices)
    all_basket_ids = []
    all_prices = []
    all_categories = []

    for i, (bid, flag) in enumerate(zip(basket_ids, fraud_flag)):
        n = PRODUCTS_PER_BASKET
        if flag:
            prices = rng.lognormal(4.0, 0.8, n).round(2)
        else:
            prices = rng.lognormal(3.0, 0.6, n).round(2)
        prices = np.clip(prices, 1.0, 500.0)
        cats = rng.choice(CATEGORIES, n)
        all_basket_ids.extend([bid] * n)
        all_prices.extend(prices)
        all_categories.extend(cats)

    products_df = pd.DataFrame({
        "basket_ID": all_basket_ids,
        "cash_price": all_prices,
        "product_category": all_categories,
    })

    return baskets_df, products_df


if __name__ == "__main__":
    baskets_df, products_df = make_data()
    baskets_path = os.path.join(OUT_DIR, "baskets.csv")
    products_path = os.path.join(OUT_DIR, "products.csv")
    baskets_df.to_csv(baskets_path, index=False)
    products_df.to_csv(products_path, index=False)
    print(f"Wrote {len(baskets_df)} baskets to {baskets_path}")
    print(f"Wrote {len(products_df)} products to {products_path}")
    print(f"  Fraud rate: {baskets_df['fraud_flag'].mean():.3f}")
