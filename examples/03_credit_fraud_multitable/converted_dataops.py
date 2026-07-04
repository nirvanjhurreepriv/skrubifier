"""
Converted from examples/03_credit_fraud_multitable/source_pipeline.py

This is the case the AST fallback analyzer flags via `source_snippet`
(build_features has no sklearn Pipeline object to introspect) — the
converter had to re-derive the 3 aggregation ops + 2 merges from the
literal pandas code.

Design choice (discovered during dynamic validation):
skrub.AggJoiner is a sklearn-style transformer: it FITS the aggregation on
the training products table and APPLIES the cached aggregates at predict time.
That means new baskets whose products are only in the test set get NaN (0.5 AUC).

This is correct behavior for some production scenarios (serve aggregates from
training time), but the original source pipeline re-computes aggregations
fresh from whatever products are passed at each call. To match that semantics,
we instead express the join as a stateless `.skb.apply_func()` on a two-DataOp
expression: products.skb.apply_func(agg_func) produces the aggregated table as
a DataOp, then baskets.merge(agg_dataop, ...) joins them — both re-evaluated
fresh at every fit/predict call (the DataOps merge operator tracks both inputs).
"""
import pandas as pd
try:
    import stratum as skrub  # drop-in accelerated backend, same DataOps API
except ImportError:
    import skrub
from sklearn.ensemble import HistGradientBoostingClassifier

baskets_df = pd.read_csv("baskets.csv")
products_df = pd.read_csv("products.csv")

products = skrub.var("products", products_df)
baskets = skrub.var("baskets", baskets_df)


def _agg_products(products_df):
    price_stats = (
        products_df.groupby("basket_ID")["cash_price"]
        .agg(["mean", "max", "sum"])
        .reset_index()
        .rename(columns={"mean": "cash_price_mean", "max": "cash_price_max",
                         "sum": "cash_price_sum"})
    )
    counts = (
        products_df.groupby("basket_ID")
        .size()
        .rename("basket_ID_count")
        .reset_index()
    )
    return price_stats.merge(counts, on="basket_ID")


# Aggregate products freshly at every fit/predict call via apply_func (stateless)
agg = products.skb.apply_func(_agg_products)

# Merge with the baskets DataOp using the DataOps .merge() operator
# (standard pandas ops on DataOps are tracked as graph edges automatically)
merged = baskets.merge(agg, on="basket_ID", how="left")

DERIVED_COLS = ["cash_price_mean", "cash_price_max", "cash_price_sum", "basket_ID_count"]
X = merged[DERIVED_COLS].skb.mark_as_X()
y = merged["fraud_flag"].skb.mark_as_y()

vectorized = X.skb.apply(skrub.TableVectorizer())
pred = vectorized.skb.apply(HistGradientBoostingClassifier(random_state=0), y=y)

learner = pred.skb.make_learner()
scores = pred.skb.cross_validate(cv=5, scoring="roc_auc")
