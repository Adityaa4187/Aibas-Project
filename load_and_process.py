# 03_load_and_preprocess.py
import os
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# =========================
# SETTINGS
# =========================
OUT_DIR = "out"
ARTIFACT_DIR = "artifacts"

TRAIN_PATH = os.path.join(OUT_DIR, "training_data.csv")
TEST_PATH = os.path.join(OUT_DIR, "test_data.csv")
ACT_PATH = os.path.join(OUT_DIR, "activation_data.csv")

TARGET = "Attrition"

PREPROCESSOR_PATH = os.path.join(ARTIFACT_DIR, "preprocessor.pkl")


def load_and_preprocess():
    os.makedirs(ARTIFACT_DIR, exist_ok=True)

    # -------------------------
    # 1) Load datasets
    # -------------------------
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    activation_df = pd.read_csv(ACT_PATH)

    # -------------------------
    # 2) Create X/y
    # -------------------------
    X_train = train_df.drop(columns=[TARGET])
    y_train = train_df[TARGET].astype(int)

    X_test = test_df.drop(columns=[TARGET])
    y_test = test_df[TARGET].astype(int)

    X_activation = activation_df.copy()   # activation has no target

    # -------------------------
    # 3) Identify numeric and categorical columns
    # -------------------------
    num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X_train.select_dtypes(include=["object"]).columns.tolist()

    print("[INFO] Numeric cols:", len(num_cols))
    print("[INFO] Categorical cols:", len(cat_cols))

    # -------------------------
    # 4) Build preprocessing pipeline
    # -------------------------
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
        ],
        remainder="drop"
    )

    # -------------------------
    # 5) Fit ONLY on training
    # -------------------------
    X_train_enc = preprocessor.fit_transform(X_train)

    # -------------------------
    # 6) Transform test + activation using same preprocessor
    # -------------------------
    X_test_enc = preprocessor.transform(X_test)
    X_activation_enc = preprocessor.transform(X_activation)

    # -------------------------
    # 7) Save preprocessor
    # -------------------------
    joblib.dump(preprocessor, PREPROCESSOR_PATH)
    print(f"[SAVED] Preprocessor → {PREPROCESSOR_PATH}")

    # -------------------------
    # 8) Return everything model-ready
    # -------------------------
    print("[INFO] Encoded train shape:", X_train_enc.shape)
    print("[INFO] Encoded test shape:", X_test_enc.shape)
    print("[INFO] Encoded activation shape:", X_activation_enc.shape)

    return X_train_enc, y_train, X_test_enc, y_test, X_activation_enc


if __name__ == "__main__":
    load_and_preprocess()
