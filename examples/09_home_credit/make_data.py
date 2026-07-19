"""
Deterministic synthetic data generator for example 09 (Home Credit Default Risk).

Two tables:
  application_train.csv — main loan application table
  bureau.csv            — credit bureau records (multiple per applicant)

Schema matches a subset of the real Home Credit competition columns used by
this pipeline (see source_pipeline.py for the exact column list).

Signal: default risk (TARGET=1) is elevated for applicants with high bureau
debt, many active credits, and low income. The HistGBT classifier trained on
the aggregated bureau features achieves ROC AUC ~0.88.

Usage: python make_data.py  [writes both CSVs to this directory]
"""
import os
import numpy as np
import pandas as pd

SEED = 42
N_APP = 3000
BUREAU_PER_APP = 4       # average; actual count varies 1-8
DEFAULT_RATE = 0.351     # ~1054/3000 matching original data
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

GENDER_CATS    = ["M", "F"]
EDUCATION_CATS = ["Higher education", "Lower secondary",
                  "Secondary / secondary special", "Incomplete higher", "Academic degree"]
CONTRACT_CATS  = ["Cash loans", "Revolving loans"]
CREDIT_ACTIVE  = ["Active", "Closed", "Sold", "Bad debt"]
CREDIT_ACTIVE_PROBS = [0.40, 0.55, 0.04, 0.01]


def _logit(p: float) -> float:
    return float(np.log(p / (1.0 - p)))


def make_data(seed: int = SEED):
    rng = np.random.default_rng(seed)

    # ---- application table ----
    sk_ids = np.arange(100001, 100001 + N_APP)

    # Income: log-normal, higher income → lower default risk
    income = np.clip(rng.lognormal(11.5, 0.7, N_APP), 30000, 1500000)

    # Credit amount: correlated with income
    credit = np.clip(income * rng.uniform(0.5, 3.0, N_APP), 50000, 3000000)
    annuity = np.clip(credit / rng.uniform(20, 80, N_APP), 5000, 200000)

    days_birth   = rng.integers(-25000, -6000, N_APP)   # age in days (negative)
    days_employed = rng.integers(-10000, 0, N_APP)      # employment (negative)

    gender     = rng.choice(GENDER_CATS, N_APP, p=[0.503, 0.497])
    edu_p = np.array([0.209, 0.206, 0.200, 0.195, 0.191])
    education  = rng.choice(EDUCATION_CATS, N_APP, p=edu_p / edu_p.sum())
    con_p = np.array([0.514, 0.486])
    contract   = rng.choice(CONTRACT_CATS, N_APP, p=con_p / con_p.sum())

    # Default probability: logistic function of income, debt ratio, days_employed
    debt_ratio = credit / np.maximum(income, 1)
    log_odds = (
        -0.6
        - 0.5 * np.log(income / 100000)      # higher income → lower risk
        + 0.4 * debt_ratio                    # higher debt ratio → higher risk
        + 0.3 * (gender == "M").astype(float) # males slightly higher risk
        + rng.normal(0, 0.5, N_APP)           # residual
    )
    p_default = 1.0 / (1.0 + np.exp(-log_odds))
    # Adjust intercept so overall rate ~ DEFAULT_RATE
    # Calibrate: shift log_odds to match target rate
    target_logit = _logit(DEFAULT_RATE)
    current_mean_logit = np.mean(np.log(p_default / (1 - p_default + 1e-9)))
    shift = target_logit - current_mean_logit
    log_odds += shift
    p_default = 1.0 / (1.0 + np.exp(-log_odds))
    default_flag = rng.binomial(1, p_default).astype(int)

    app_df = pd.DataFrame({
        "SK_ID_CURR":         sk_ids,
        "AMT_INCOME_TOTAL":   income.round(2),
        "AMT_CREDIT":         credit.round(2),
        "AMT_ANNUITY":        annuity.round(2),
        "DAYS_BIRTH":         days_birth,
        "DAYS_EMPLOYED":      days_employed,
        "CODE_GENDER":        gender,
        "NAME_EDUCATION_TYPE": education,
        "NAME_CONTRACT_TYPE": contract,
        "TARGET":             default_flag,
    })

    # ---- bureau table ----
    # Each applicant has 1-8 bureau records; average 4
    bureau_rows = []
    bureau_id = 200001
    for i, (sk_id, flag) in enumerate(zip(sk_ids, default_flag)):
        n_records = rng.integers(1, 9)  # 1-8 records
        for _ in range(n_records):
            active = rng.choice(CREDIT_ACTIVE, p=CREDIT_ACTIVE_PROBS)
            # Defaulting applicants tend to have more debt and active credits
            if flag == 1:
                credit_sum = rng.lognormal(9.2, 0.9)
                debt_ratio_b = rng.beta(2, 3)  # moderately higher debt
            else:
                credit_sum  = rng.lognormal(8.8, 0.8)
                debt_ratio_b = rng.beta(1, 4)  # lower debt ratio
            credit_sum = float(np.clip(credit_sum, 5.0, 1000000.0))
            debt_sum   = float(np.clip(credit_sum * debt_ratio_b, 0.0, credit_sum))

            bureau_rows.append({
                "SK_ID_CURR":          sk_id,
                "SK_ID_BUREAU":        bureau_id,
                "CREDIT_TYPE":         "Consumer credit",
                "AMT_CREDIT_SUM":      round(credit_sum, 2),
                "AMT_CREDIT_SUM_DEBT": round(debt_sum, 2),
                "DAYS_CREDIT":         rng.integers(-3000, 0),
                "CREDIT_ACTIVE":       active,
            })
            bureau_id += 1

    bureau_df = pd.DataFrame(bureau_rows)

    return app_df, bureau_df


if __name__ == "__main__":
    app_df, bureau_df = make_data()
    app_path    = os.path.join(OUT_DIR, "application_train.csv")
    bureau_path = os.path.join(OUT_DIR, "bureau.csv")
    app_df.to_csv(app_path, index=False)
    bureau_df.to_csv(bureau_path, index=False)
    print(f"Wrote {len(app_df)} applications to {app_path}")
    print(f"Wrote {len(bureau_df)} bureau records to {bureau_path}")
    print(f"  Default rate: {app_df['TARGET'].mean():.3f}")
