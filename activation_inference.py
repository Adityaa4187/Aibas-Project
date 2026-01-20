# activation_inference.py
# Blind Testing Script
# Loads activation_data.csv (1 row), applies preprocessor, runs all 4 models,
# prints predictions + saves report.

import os
import joblib
import numpy as np
import pandas as pd
import statsmodels.api as sm
import tensorflow as tf

from early_risk import compute_early_risk

# -------------------------
# PATHS
# -------------------------
ACT_PATH = os.path.join("out", "activation_data.csv")
PREPROCESSOR_PATH = os.path.join("artifacts", "preprocessor.pkl")

OLS_PATH = os.path.join("models", "ols_model.pkl")
LOGREG_PATH = os.path.join("models", "logreg_model.pkl")
RF_PATH = os.path.join("models", "rf_model.pkl")
ANN_PATH = os.path.join("models", "ann_model.h5")

REPORTS_DIR = "reports"
OUT_REPORT = os.path.join(REPORTS_DIR, "activation_risk_report.csv")

# Optional thresholds (if you create later, else fallback to 0.5)
THRESH_DIR = "thresholds"
DEFAULT_THRESHOLD = 0.5


# -------------------------
# HELPERS
# -------------------------
def load_threshold(model_name: str):
    """
    Loads saved threshold if exists else returns default.
    Expects file: thresholds/<model_name>_threshold.txt
    """
    path = os.path.join(THRESH_DIR, f"{model_name}_threshold.txt")
    if os.path.exists(path):
        with open(path, "r") as f:
            val = float(f.read().strip())
        return val
    return DEFAULT_THRESHOLD


def predict_proba_ols(model, X_enc):
    """
    statsmodels OLS predict: expects dense + constant
    """
    X_dense = X_enc.toarray() if hasattr(X_enc, "toarray") else X_enc
    X_dense = sm.add_constant(X_dense, has_constant="add").astype(float)
    probs = model.predict(X_dense)
    probs = np.clip(probs, 0, 1)
    return float(probs[0])


def risk_bucket_from_prob(p: float):
    """
    Simple probability-based risk bucket.
    """
    if p < 0.33:
        return "Low"
    elif p < 0.66:
        return "Medium"
    return "High"


def run_activation_inference():
    os.makedirs(REPORTS_DIR, exist_ok=True)

    # 1) Load activation row
    act_df = pd.read_csv(ACT_PATH)
    if len(act_df) != 1:
        print(f"[WARNING] activation_data.csv has {len(act_df)} rows, expected 1. Using first row only.")
        act_df = act_df.iloc[[0]].copy()

    # Compute early risk (rule-based) from raw activation row
    early_score, early_bucket, early_reason = compute_early_risk(act_df.iloc[0])

    print("\n===== EARLY ATTRITION RISK (RULE-BASED) =====")
    print(f"Score: {early_score}/100")
    print(f"Bucket: {early_bucket}")
    print(f"Reason: {early_reason}\n")

    # 2) Load preprocessor + transform
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    X_act_enc = preprocessor.transform(act_df)

    # 3) Load models
    ols_model = joblib.load(OLS_PATH)
    logreg_model = joblib.load(LOGREG_PATH)
    rf_model = joblib.load(RF_PATH)
    ann_model = tf.keras.models.load_model(ANN_PATH)

    # -------------------------
    # 4) Predict probabilities
    # -------------------------
    probs = {}

    # OLS
    probs["ols"] = predict_proba_ols(ols_model, X_act_enc)

    # Logistic Regression
    probs["logreg"] = float(logreg_model.predict_proba(X_act_enc)[:, 1][0])

    # Random Forest
    probs["rf"] = float(rf_model.predict_proba(X_act_enc)[:, 1][0])

    # ANN (needs dense)
    X_dense = X_act_enc.toarray() if hasattr(X_act_enc, "toarray") else X_act_enc
    probs["ann"] = float(ann_model.predict(X_dense).ravel()[0])


    # 5) Apply thresholds + build report
    rows = []
    print("\nACTIVATION BLIND TEST\n")

    for model_name, p in probs.items():
        thr = load_threshold(model_name)
        pred_class = 1 if p >= thr else 0

        decision = "LEAVE (Attrition=1)" if pred_class == 1 else "STAY (Attrition=0)"
        risk = risk_bucket_from_prob(p)

        print(f"--- {model_name.upper()} ---")
        print(f"P(Leave) = {p:.4f}")
        print(f"Threshold = {thr:.2f}")
        print(f"Prediction = {decision}")
        print(f"Risk Bucket = {risk}")
        print(f"Early Risk = {early_bucket} (Score={early_score})\n")

        rows.append({
            "model": model_name,
            "prob_leave": round(p, 6),
            "threshold_used": thr,
            "predicted_class": pred_class,
            "decision": "Leave" if pred_class == 1 else "Stay",
            "risk_bucket": risk,

            # Early risk (same for all models since based on employee features)
            "early_risk_score": early_score,
            "early_risk_bucket": early_bucket,
            "early_risk_reason": early_reason
        })

    report_df = pd.DataFrame(rows)
    report_df.to_csv(OUT_REPORT, index=False)

    print(f"[SAVED] Activation blind-test report → {OUT_REPORT}")
    return report_df


if __name__ == "__main__":
    run_activation_inference()
