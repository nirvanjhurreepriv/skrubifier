try:
    import stratum as skrub
except ImportError:
    import skrub

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

# Load data upfront for schema inference (required by skrub.var)
baskets_df = pd.read_csv("baskets.csv")
products_df = pd.read_csv("products.csv")

# Declare pipeline inputs
baskets = skrub.var("baskets", baskets_df)
products = skrub.var("products", products_df)

# --- Re-implement build_features logic using DataOps ---
# The original function is stateless pandas operations, so we use .skb.apply_func

def _agg_products(df):
    agg = (
        df.groupby("basket_ID")["cash_price"]
        .agg(["mean", "max", "sum"])
        .rename(columns={"mean": "price_mean", "max": "price_max", "sum": "price_sum"})
        .reset_index()
    )
    return agg

def _count_items(df):
    n_items = df.groupby("basket_ID").size().rename("n_items").reset_index()
    return n_items

# Create aggregation nodes
agg_price = products.skb.apply_func(_agg_products)
agg_count = products.skb.apply_func(_count_items)

# Perform joins
merged = baskets.merge(agg_price, on="basket_ID", how="left")
merged = merged.merge(agg_count, on="basket_ID", how="left")

# --- Define X, y and fit the model ---
FEATURE_COLS = ["price_mean", "price_max", "price_sum", "n_items"]

X = merged[FEATURE_COLS].skb.mark_as_X()
y = merged["fraud_flag"].skb.mark_as_y()

# HistGradientBoostingClassifier handles numeric data well directly
pred = X.skb.apply(HistGradientBoostingClassifier(random_state=0), y=y)

learner = pred.skb.make_learner()