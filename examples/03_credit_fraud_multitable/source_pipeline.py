"""Representative of MLE-Bench multi-table competitions (e.g. IEEE-CIS
fraud, Home Credit): a main table joined with an aggregated auxiliary
table via manual pandas groupby/agg/merge — the pattern skrub's AggJoiner
was purpose-built to replace."""
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

TASK = "classification"
TARGET_COLUMN = "fraud_flag"


def build_features(baskets_df: pd.DataFrame, products_df: pd.DataFrame) -> pd.DataFrame:
    agg = (
        products_df.groupby("basket_ID")["cash_price"]
        .agg(["mean", "max", "sum"])
        .rename(columns={"mean": "price_mean", "max": "price_max", "sum": "price_sum"})
        .reset_index()
    )
    n_items = products_df.groupby("basket_ID").size().rename("n_items").reset_index()
    merged = baskets_df.merge(agg, on="basket_ID", how="left").merge(n_items, on="basket_ID", how="left")
    return merged


FEATURE_COLUMNS = ["price_mean", "price_max", "price_sum", "n_items"]

if __name__ == "__main__":
    baskets_df = pd.read_csv("baskets.csv")
    products_df = pd.read_csv("products.csv")
    merged = build_features(baskets_df, products_df)
    X, y = merged[FEATURE_COLUMNS], merged[TARGET_COLUMN]
    model = HistGradientBoostingClassifier(random_state=0)
    model.fit(X, y)
