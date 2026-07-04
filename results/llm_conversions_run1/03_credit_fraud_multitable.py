try:
    import stratum as skrub
except ImportError:
    import skrub

from sklearn.ensemble import HistGradientBoostingClassifier

# Declare pipeline inputs (sample values are placeholders - actual data loaded at runtime)
baskets_df = skrub.var("baskets", None)
products_df = skrub.var("products", None)

# Re-implement build_features using AggJoiner instead of manual pandas groupby+merge
# The original does: 
#   agg = products_df.groupby("basket_ID")["cash_price"].agg(["mean", "max", "sum"]).rename(...)
#   n_items = products_df.groupby("basket_ID").size().rename("n_items")
#   merged = baskets_df.merge(agg, on="basket_ID").merge(n_items, on="basket_ID")
# This can be done with TWO AggJoiner calls (one for the price stats, one for count) 
# OR ONE AggJoiner call that computes both price aggregates AND count using 'count' operation.

# Strategy: Use ONE AggJoiner that computes price stats + count via operations list
baskets_with_features = baskets_df.skb.apply(
    skrub.AggJoiner(
        aux_table=products_df,
        aux_key="basket_ID",
        main_key="basket_ID",
        cols=["cash_price"],
        operations=["mean", "max", "sum", "count"],
    )
)

# At this point, the result has columns: 
#   cash_price_mean, cash_price_max, cash_price_sum, cash_price_count 
# We need to rename cash_price_count -> n_items, and rename others to match original
renamed = baskets_with_features.assign(
    n_items=baskets_with_features["cash_price_count"],
    price_mean=baskets_with_features["cash_price_mean"],
    price_max=baskets_with_features["cash_price_max"],
    price_sum=baskets_with_features["cash_price_sum"]
).drop(columns=["cash_price_mean", "cash_price_max", "cash_price_sum", "cash_price_count"])

# Drop non-feature columns (keep only features + target if present)
# Since target column is "target" in IR, but original code uses "fraud_flag", 
# use "fraud_flag" as it appears in the original source and is more specific.
TARGET_COLUMN = "fraud_flag"
# In practice, the target column may not be in baskets_with_features if it's only in one table
# We'll assume TARGET_COLUMN exists and mark it appropriately
X = renamed.skb.select(["price_mean", "price_max", "price_sum", "n_items"]).skb.mark_as_X()
y = renamed[TARGET_COLUMN].skb.mark_as_y()

# Apply estimator
pred = X.skb.apply(HistGradientBoostingClassifier(random_state=0), y=y)

# Export learner
learner = pred.skb.make_learner()

# If cross-validation is needed (per task description)
# scores = pred.skb.cross_validate(cv=5, scoring="roc_auc")