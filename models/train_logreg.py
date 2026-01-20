# models/train_logreg.py
import os
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report
)

from plots import plot_confusion_matrix, plot_roc_curve, plot_pr_curve

MODEL_PATH = os.path.join("models", "logreg_model.pkl")
PLOT_DIR = os.path.join("reports", "model_plots", "logreg")


def train_logreg(X_train, y_train):
    model = LogisticRegression(
        max_iter=3000,
        class_weight="balanced",
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    joblib.dump(model, MODEL_PATH)
    print(f"[SAVED] Logistic Regression -> {MODEL_PATH}")
    return model


def evaluate_logreg(model, X_test, y_test, threshold=0.5):
    probs = model.predict_proba(X_test)[:, 1]

    roc = roc_auc_score(y_test, probs)
    pr = average_precision_score(y_test, probs)

    y_pred = (probs >= threshold).astype(int)

    cm = confusion_matrix(y_test, y_pred)

    print("\n LOGISTIC REGRESSION RESULTS")
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
        title="Logistic Regression - Confusion Matrix",
        save_path=os.path.join(PLOT_DIR, "confusion_matrix.png")
    )

    plot_roc_curve(
        y_test, probs,
        title="Logistic Regression - ROC Curve",
        save_path=os.path.join(PLOT_DIR, "roc_curve.png")
    )

    plot_pr_curve(
        y_test, probs,
        title="Logistic Regression - PR Curve",
        save_path=os.path.join(PLOT_DIR, "pr_curve.png")
    )


    return probs
