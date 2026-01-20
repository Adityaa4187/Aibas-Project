# 02_split_step1_dataset.py
import os
import pandas as pd
from sklearn.model_selection import train_test_split

# =========================
# SETTINGS
# =========================
STEP1_PATH = os.path.join("data", "step1", "step1_cleaned_dataset.csv")
OUT_DIR = "out"

TARGET = "Attrition"
SEED = 42
TEST_SIZE = 0.20

TRAIN_OUT = os.path.join(OUT_DIR, "training_data.csv")
TEST_OUT = os.path.join(OUT_DIR, "test_data.csv")
ACTIVATION_OUT = os.path.join(OUT_DIR, "activation_data.csv")


def split_dataset():
    os.makedirs(OUT_DIR, exist_ok=True)

    # 1) Load step1 dataset
    df = pd.read_csv(STEP1_PATH)

    if TARGET not in df.columns:
        raise ValueError(f"[ERROR] Target column '{TARGET}' not found in step1 dataset!")

    # 2) Official train/test split (frozen)
    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=df[TARGET]
    )

    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    # 3) Save split datasets
    train_df.to_csv(TRAIN_OUT, index=False)
    test_df.to_csv(TEST_OUT, index=False)

    print(f"[SAVED] {TRAIN_OUT}  shape={train_df.shape}")
    print(f"[SAVED] {TEST_OUT}   shape={test_df.shape}")

    # 4) Activation dataset (1 random row from test, blind)
    activation_row = test_df.sample(n=1, random_state=SEED)
    activation_x = activation_row.drop(columns=[TARGET])

    activation_x.to_csv(ACTIVATION_OUT, index=False)

    print(f"[SAVED] {ACTIVATION_OUT} shape={activation_x.shape}")
    print("[INFO] Activation row index (from test):", activation_row.index.tolist())


if __name__ == "__main__":
    split_dataset()
