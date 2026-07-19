try:
    import stratum as skrub
except ImportError:
    import skrub
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

baskets_df = pd.read_csv("baskets.csv")
products_df = pd.read_csv("products.csv")

baskets = skrub.var("baskets", baskets_df)
products = skrub.var("products", products_df)

def _agg_products(df):
    return (df.groupby("basket_ID")["cash_price"]
              .agg(["mean", "max", "sum"])
              .rename(columns={"mean": "price_mean", "max": "price_max", "sum": "price_sum"})
              .reset_index())

def _n_items(df):
    return (df.groupby("basket_ID").size()
              .rename("n_items")
              .reset_index())

agg = products.skb.apply_func(_agg_products)
n_items = products.skb.apply_func(_n_items)
merged = baskets.merge(agg, on="basket_ID", how="left").merge(n_items, on="basket_ID", how="left")

FEATURE_COLUMNS = ["price_mean", "price_max", "price_sum", "n_items"]
X = merged[FEATURE_COLUMNS].skb.mark_as_X()
y = merged["fraud_flag"].skb.mark_as_y()

vectorized = X.skb.apply(skrub.TableVectorizer())
pred = vectorized.skb.apply(HistGradientBoostingClassifier(random_state=0), y=y)

learner = pred.skb.make_learner()