try:
    import stratum as skrub
except ImportError:
    import skrub
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

TARGET_COLUMN = "TARGET"

# Load real DataFrames first (required by skrub DataOps)
app_df = pd.read_csv("application_train.csv")
bureau_df = pd.read_csv("bureau.csv")

# Declare pipeline inputs (must pass real DataFrames, not None)
app = skrub.var("app", app_df)
bureau = skrub.var("bureau", bureau_df)

# Define aggregation function as stateless operation
def _aggregate_bureau(bureau_df):
    return (bureau_df.groupby("SK_ID_CURR")
              .agg(
                  bureau_credit_sum=("AMT_CREDIT_SUM", "sum"),
                  bureau_debt_mean=("AMT_CREDIT_SUM_DEBT", "mean"),
                  bureau_count=("SK_ID_BUREAU", "count"),
                  bureau_active_count=("CREDIT_ACTIVE", lambda x: (x == "Active").sum()),
              )
              .reset_index())

# Apply aggregation using apply_func (stateless, re-runs at every call)
bureau_agg = bureau.skb.apply_func(_aggregate_bureau)

# Merge onto main table
merged = app.merge(bureau_agg, on="SK_ID_CURR", how="left")

# Encode categorical columns as int codes (stateless transformation)
# Use apply_func to handle the in-place assignment to each column
def _encode_categories(df):
    for col in ["CODE_GENDER", "NAME_EDUCATION_TYPE", "NAME_CONTRACT_TYPE"]:
        df[col] = df[col].astype("category").cat.codes
    return df

merged = merged.skb.apply_func(_encode_categories)

# Select features and mark as X/y
FEATURE_COLUMNS = [
    "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY",
    "DAYS_BIRTH", "DAYS_EMPLOYED",
    "CODE_GENDER", "NAME_EDUCATION_TYPE", "NAME_CONTRACT_TYPE",
    "bureau_credit_sum", "bureau_debt_mean", "bureau_count", "bureau_active_count"
]

X = merged[FEATURE_COLUMNS].skb.mark_as_X()
y = merged[TARGET_COLUMN].skb.mark_as_y()

# Apply TableVectorizer to handle encoding of remaining categoricals and numeric features
vectorized = X.skb.apply(skrub.TableVectorizer())

# Apply estimator (HistGradientBoostingClassifier) using apply()
pred = vectorized.skb.apply(HistGradientBoostingClassifier(random_state=0), y=y)

# Export as a SkrubLearner
learner = pred.skb.make_learner()

# Optionally, run cross-validation (required for evaluation)
scores = pred.skb.cross_validate(cv=5, scoring="roc_auc")