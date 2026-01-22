# preprocessing_step1.py
import os
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

from plots import plot_numeric_corr, plot_cramers_v

RAW_PATH = os.path.join("data", "raw", "WA_Fn-UseC_-HR-Employee-Attrition.csv")
STEP1_DIR = os.path.join("data", "step1")
STEP1_OUT = os.path.join(STEP1_DIR, "step1_cleaned_dataset.csv")
PLOTS_DIR = os.path.join("reports", "step1_plots")

TARGET = "Attrition"

# thresholds
NUM_CORR_THRESHOLD = 0.02
CRAMERSV_THRESHOLD = 0.05


def cramers_v(x, y):
    table = pd.crosstab(x, y)
    if table.shape[0] <= 1 or table.shape[1] <= 1:
        return 0.0

    chi2 = chi2_contingency(table)[0]
    n = table.values.sum()
    r, k = table.shape
    return float(np.sqrt((chi2 / n) / (min(r - 1, k - 1) + 1e-10)))


def detect_id_like_columns(df):
    id_like = []
    for col in df.columns:
        if col == TARGET:
            continue

        col_lower = col.lower()

        if any(key in col_lower for key in ["id", "number"]):
            id_like.append(col)
            continue

        if df[col].nunique(dropna=False) / len(df) > 0.98:
            id_like.append(col)

    return sorted(set(id_like))


def detect_constant_columns(df):
    const = []
    for col in df.columns:
        if col != TARGET and df[col].nunique(dropna=False) <= 1:
            const.append(col)
    return sorted(const)


def run_step1_preprocessing():
    df = pd.read_csv(RAW_PATH)

    # target mapping
    df[TARGET] = df[TARGET].map({"Yes": 1, "No": 0})
    if df[TARGET].isna().any():
        bad_vals = df.loc[df[TARGET].isna(), TARGET].unique()
        raise ValueError(f"[ERROR] Unexpected Attrition values found: {bad_vals}")
    df[TARGET] = df[TARGET].astype(int)

    print("[INFO] Target distribution:", df[TARGET].value_counts().to_dict())

    # columns to remove (constant + id-like)
    constant_cols = detect_constant_columns(df)
    id_like_cols = detect_id_like_columns(df)

    print("\n[INFO] Constant columns detected:", constant_cols)
    print("[INFO] ID-like columns detected:", id_like_cols)

    # numeric correlation check (FIXED)
    num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != TARGET]

    corr = df[num_cols + [TARGET]].corr(numeric_only=True)[TARGET].drop(TARGET)
    corr_sorted = corr.reindex(corr.abs().sort_values(ascending=False).index)

    low_corr_cols = corr_sorted[corr_sorted.abs() < NUM_CORR_THRESHOLD].index.tolist()
    print(f"\n[INFO] Numeric cols suggested drop (|corr| < {NUM_CORR_THRESHOLD}):")
    print(low_corr_cols)

    plot_numeric_corr(corr_sorted, PLOTS_DIR)

    # categorical cramers-v check
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    cramer_series = pd.Series(
        {c: cramers_v(df[c], df[TARGET]) for c in cat_cols}
    ).sort_values(ascending=False)

    low_cat_cols = cramer_series[cramer_series < CRAMERSV_THRESHOLD].index.tolist()
    print(f"\n[INFO] Categorical cols suggested drop (V < {CRAMERSV_THRESHOLD}):")
    print(low_cat_cols)

    plot_cramers_v(cramer_series, PLOTS_DIR)

    # final drop + save
    drop_cols = sorted(set(constant_cols + id_like_cols + low_corr_cols + low_cat_cols))
    drop_cols = [c for c in drop_cols if c != TARGET]  # never drop target

    df_clean = df.drop(columns=drop_cols, errors="ignore").copy()

    os.makedirs(STEP1_DIR, exist_ok=True)
    df_clean.to_csv(STEP1_OUT, index=False)

    print("\n[SUCCESS] Step-1 cleaned dataset saved.")
    print("[INFO] Dropped columns count:", len(drop_cols))
    print("[INFO] Remaining columns:", df_clean.shape[1])
    print("[INFO] Saved to:", STEP1_OUT)


if __name__ == "__main__":
    run_step1_preprocessing()
