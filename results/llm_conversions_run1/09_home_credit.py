try:
    import stratum as skrub
except ImportError:
    import skrub

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

# Declare pipeline inputs (the two source tables)
app_df = skrub.var("app", app_df)
bureau_df = skrub.var("bureau", bureau_df)

# Step 1: Aggregate bureau data using AggJoiner (replaces manual groupby+agg+merge)
bureau_agg = bureau_df.groupby("SK_ID_CURR").agg(
    bureau_credit_sum=("AMT_CREDIT_SUM", "sum"),
    bureau_debt_mean=("AMT_CREDIT_SUM_DEBT", "mean"),
    bureau_count=("SK_ID_BUREAU", "count"),
    bureau_active_count=("CREDIT_ACTIVE", lambda x: (x == "Active").sum()),
).reset_index()

# Use AggJoiner for the join operation (main table + bureau aggregates)
app_with_bureau = app_df.skb.apply(
    skrub.AggJoiner(
        aux_table=bureau_agg,
        aux_key="SK_ID_CURR",
        main_key="SK_ID_CURR",
        cols=["AMT_CREDIT_SUM", "AMT_CREDIT_SUM_DEBT", "SK_ID_BUREAU", "CREDIT_ACTIVE"],
        operations=["sum", "mean", "count", lambda x: (x == "Active").sum()],
    ),
)

# Encode categorical columns as int codes (stateless transformation)
categorical_cols = ["CODE_GENDER", "NAME_EDUCATION_TYPE", "NAME_CONTRACT_TYPE"]
for col in categorical_cols:
    app_with_bureau[col] = app_with_bureau[col].skb.apply_func(
        lambda s: s.astype("category").cat.codes
    )

# Select feature columns and target
FEATURE_COLUMNS = [
    "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY",
    "DAYS_BIRTH", "DAYS_EMPLOYED",
    "CODE_GENDER", "NAME_EDUCATION_TYPE", "NAME_CONTRACT_TYPE",
    "bureau_credit_sum", "bureau_debt_mean", "bureau_count", "bureau_active_count"
]
TARGET_COLUMN = "TARGET"

X = app_with_bureau.skb.select(skrub.selectors.cols(*FEATURE_COLUMNS)).skb.mark_as_X()
y = app_with_bureau[TARGET_COLUMN].skb.mark_as_y()

# Apply HistGradientBoostingClassifier
model = HistGradientBoostingClassifier(random_state=0)
pred = X.skb.apply(model, y=y)

# Export the learner
learner = pred.skb.make_learner()

# Cross-validation (if needed for evaluation)
cv_scores = pred.skb.cross_validate(cv=5, scoring="roc_auc")