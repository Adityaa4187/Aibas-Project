# models/train_ols.py
import os
import joblib
import numpy as np
import statsmodels.api as sm

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report
)

from plots import plot_confusion_matrix, plot_roc_curve, plot_pr_curve

MODEL_PATH = os.path.join("models", "ols_model.pkl")
PLOT_DIR = os.path.join("reports", "model_plots", "ols")


def train_ols(X_train, y_train):
    """
    OLS baseline: statsmodels regression
    """
    X_train_dense = X_train.toarray() if hasattr(X_train, "toarray") else X_train
    X_train_dense = sm.add_constant(X_train_dense, has_constant="add").astype(float)

    model = sm.OLS(y_train, X_train_dense).fit()
    joblib.dump(model, MODEL_PATH)

    print(f"[SAVED] OLS model -> {MODEL_PATH}")
    return model


def predict_proba_ols(model, X):
    X_dense = X.toarray() if hasattr(X, "toarray") else X
    X_dense = sm.add_constant(X_dense, has_constant="add").astype(float)
    probs = model.predict(X_dense)
    probs = np.clip(probs, 0, 1)  # OLS can exceed [0,1]
    return probs


def evaluate_ols(model, X_test, y_test, threshold=0.5):
    probs = predict_proba_ols(model, X_test)

    roc = roc_auc_score(y_test, probs)
    pr = average_precision_score(y_test, probs)

    y_pred = (probs >= threshold).astype(int)

    cm = confusion_matrix(y_test, y_pred)

    print("\n\n OLS RESULTS")
    print(f"ROC-AUC: {roc:.4f}")
    print(f"PR-AUC:  {pr:.4f}")
    print("Confusion Matrix:\n", cm)
    print("Classification Report:\n", classification_report(y_test, y_pred))

    # -------------------------
    # PLOTS (show + save)
    # -------------------------
    os.makedirs(PLOT_DIR, exist_ok=True)

    plot_confusion_matrix(
        cm,
        labels=["Stay(0)", "Leave(1)"],
        title="OLS - Confusion Matrix",
        save_path=os.path.join(PLOT_DIR, "confusion_matrix.png")
    )

    plot_roc_curve(
        y_test, probs,
        title="OLS - ROC Curve",
        save_path=os.path.join(PLOT_DIR, "roc_curve.png")
    )

    plot_pr_curve(
        y_test, probs,
        title="OLS - PR Curve",
        save_path=os.path.join(PLOT_DIR, "pr_curve.png")
    )


    return probs
