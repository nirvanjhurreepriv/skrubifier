try:
    import stratum as skrub
except ImportError:
    import skrub
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

# Load real DataFrames before wrapping in skrub.var
baskets_df = pd.read_csv("baskets.csv")
products_df = pd.read_csv("products.csv")

# Declare pipeline inputs — always pass the loaded DataFrame, never None
baskets = skrub.var("baskets", baskets_df)
products = skrub.var("products", products_df)

# Stateless aggregation function to compute aggregations fresh each time
def _agg_products(df):
    agg = (
        df.groupby("basket_ID")["cash_price"]
        .agg(["mean", "max", "sum"])
        .rename(columns={"mean": "price_mean", "max": "price_max", "sum": "price_sum"})
        .reset_index()
    )
    n_items = df.groupby("basket_ID").size().rename("n_items").reset_index()
    return agg.merge(n_items, on="basket_ID", how="left")

# Apply the stateless aggregation to the products DataOp
agg = products.skb.apply_func(_agg_products)

# Join the aggregated features to the baskets DataOp
merged = baskets.merge(agg, on="basket_ID", how="left")

# Mark features and target
FEATURE_COLUMNS = ["price_mean", "price_max", "price_sum", "n_items"]
X = merged[FEATURE_COLUMNS].skb.mark_as_X()
y = merged["fraud_flag"].skb.mark_as_y()

# Apply the estimator
vectorized = X.skb.apply(skrub.TableVectorizer())
pred = vectorized.skb.apply(HistGradientBoostingClassifier(random_state=0), y=y)

# Export as learner and run cross-validation
learner = pred.skb.make_learner()
scores = pred.skb.cross_validate(cv=5, scoring="roc_auc")