# plots.py
import os
import numpy as np
import matplotlib.pyplot as plt


def _save_show_close(save_path):
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.show()
    plt.close()
    print(f"[SAVED] {save_path}")


# EXISTING PLOTS (KEEP SAME)
def plot_numeric_corr(corr_sorted, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    plt.figure(figsize=(12, 6))
    corr_sorted.plot(kind="bar")
    plt.axhline(0, linewidth=1)
    plt.title("Numeric Feature Correlation vs Attrition")
    plt.ylabel("Pearson Correlation")
    plt.xticks(rotation=75)

    out_path = os.path.join(out_dir, "numeric_corr_bar.png")
    _save_show_close(out_path)


def plot_cramers_v(cramer_series, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    plt.figure(figsize=(12, 6))
    cramer_series.plot(kind="bar")
    plt.title("Categorical Feature Association vs Attrition (Cramér’s V)")
    plt.ylabel("Cramér’s V (0 to 1)")
    plt.xticks(rotation=75)

    out_path = os.path.join(out_dir, "categorical_cramersv_bar.png")
    _save_show_close(out_path)


# NEW MODEL PLOTS
def plot_confusion_matrix(cm, labels, title, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.figure(figsize=(5, 4))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()

    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels)
    plt.yticks(tick_marks, labels)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, str(cm[i, j]),
                ha="center",
                color="white" if cm[i, j] > thresh else "black"
            )

    plt.ylabel("Actual")
    plt.xlabel("Predicted")

    _save_show_close(save_path)


def plot_roc_curve(y_true, y_prob, title, save_path):
    from sklearn.metrics import roc_curve, auc
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"ROC-AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")

    _save_show_close(save_path)


def plot_pr_curve(y_true, y_prob, title, save_path):
    from sklearn.metrics import precision_recall_curve, average_precision_score
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)

    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, label=f"PR-AUC = {pr_auc:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.legend(loc="lower left")

    _save_show_close(save_path)
