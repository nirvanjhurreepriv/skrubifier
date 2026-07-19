try:
    import stratum as skrub
except ImportError:
    import skrub
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

# Load raw tables
baskets_df = pd.read_csv("baskets.csv")
products_df = pd.read_csv("products.csv")

# Declare pipeline inputs
baskets = skrub.var("baskets", baskets_df)
products = skrub.var("products", products_df)

# Stateless aggregation helpers
def _agg_price(df):
    return (
        df.groupby("basket_ID")["cash_price"]
          .agg(["mean", "max", "sum"])
          .rename(columns={"mean": "price_mean",
                           "max": "price_max",
                           "sum": "price_sum"})
          .reset_index()
    )

def _agg_n_items(df):
    return (
        df.groupby("basket_ID")
          .size()
          .rename("n_items")
          .reset_index()
    )

# Apply aggregations (stateless) and join back to the main table
price_agg = products.skb.apply_func(_agg_price)
n_items_agg = products.skb.apply_func(_agg_n_items)

merged = baskets.merge(price_agg, on="basket_ID", how="left") \
                .merge(n_items_agg, on="basket_ID", how="left")

# Feature / target definition
FEATURE_COLUMNS = ["price_mean", "price_max", "price_sum", "n_items"]
X = merged[FEATURE_COLUMNS].skb.mark_as_X()
y = merged["fraud_flag"].skb.mark_as_y()

# Optional vectorization (handles missing values, scaling, etc.)
vectorized = X.skb.apply(skrub.TableVectorizer())

# Model training / inference node
pred = vectorized.skb.apply(
    HistGradientBoostingClassifier(random_state=0),
    y=y,
)

# Export as a SkrubLearner
learner = pred.skb.make_learner()