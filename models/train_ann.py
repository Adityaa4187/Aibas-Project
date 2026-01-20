# models/train_ann.py
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report
)

from plots import plot_confusion_matrix, plot_roc_curve, plot_pr_curve

MODEL_PATH = os.path.join("models", "ann_model.h5")
PLOT_DIR = os.path.join("reports", "model_plots", "ann")


def build_ann(input_dim: int):
    model = tf.keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.25),
        layers.Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.AUC(name="roc_auc", curve="ROC"),
            tf.keras.metrics.AUC(name="pr_auc", curve="PR")
        ]
    )
    return model


def train_ann(X_train, y_train, epochs=25, batch_size=32):
    # ANN needs dense arrays
    X_train_dense = X_train.toarray() if hasattr(X_train, "toarray") else X_train

    model = build_ann(X_train_dense.shape[1])

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True
        )
    ]

    model.fit(
        X_train_dense, y_train,
        validation_split=0.2,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
        callbacks=callbacks
    )

    model.save(MODEL_PATH)
    print(f"[SAVED] ANN model -> {MODEL_PATH}")
    return model


def evaluate_ann(model, X_test, y_test, threshold=0.5):
    # ANN needs dense arrays
    X_test_dense = X_test.toarray() if hasattr(X_test, "toarray") else X_test

    probs = model.predict(X_test_dense).ravel()

    roc = roc_auc_score(y_test, probs)
    pr = average_precision_score(y_test, probs)

    y_pred = (probs >= threshold).astype(int)

    cm = confusion_matrix(y_test, y_pred)

    print("\n ANN RESULTS")
    print(f"ROC-AUC: {roc:.4f}")
    print(f"PR-AUC:  {pr:.4f}")
    print("Confusion Matrix:\n", cm)
    print("Classification Report:\n", classification_report(y_test, y_pred))


    # PLOTS (show + save)
    os.makedirs(PLOT_DIR, exist_ok=True)

    plot_confusion_matrix(
        cm,
        labels=["Stay(0)", "Leave(1)"],
        title="ANN - Confusion Matrix",
        save_path=os.path.join(PLOT_DIR, "confusion_matrix.png")
    )

    plot_roc_curve(
        y_test, probs,
        title="ANN - ROC Curve",
        save_path=os.path.join(PLOT_DIR, "roc_curve.png")
    )

    plot_pr_curve(
        y_test, probs,
        title="ANN - PR Curve",
        save_path=os.path.join(PLOT_DIR, "pr_curve.png")
    )

    return probs
