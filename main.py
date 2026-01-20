# main.py
# Entry point of the whole project.
# Step 1: download raw dataset from GitHub using scrapping.py
# Step 2: preprocessing step1 (mapping + auto drop useless cols + save plots)
# Step 3: split cleaned dataset into training/test/activation
# Step 4: reload and preprocess (StandardScaler + OneHotEncoder)
# Step 5: train all models (OLS, LogReg, RF, ANN) + save + plots
# Step 6: run blind activation inference

import os
import sys

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from scrapping import download_dataset
from preprocessing import run_step1_preprocessing
from splitting import split_dataset
from load_and_process import load_and_preprocess

from models.train_ols import train_ols, evaluate_ols
from models.train_logreg import train_logreg, evaluate_logreg
from models.train_rf import train_rf, evaluate_rf
from models.train_ann import train_ann, evaluate_ann

from activation_inference import run_activation_inference


def main():
    # Step 1
    download_dataset()

    # Step 2
    run_step1_preprocessing()

    # Step 3
    split_dataset()

    # Step 4
    X_train, y_train, X_test, y_test, X_activation = load_and_preprocess()

    # Step 5 - Train + Evaluate Models (with plots)
    print("\n================ TRAINING MODELS ================\n")

    # OLS
    ols_model = train_ols(X_train, y_train)
    evaluate_ols(ols_model, X_test, y_test, threshold=0.5)

    # Logistic Regression
    logreg_model = train_logreg(X_train, y_train)
    evaluate_logreg(logreg_model, X_test, y_test, threshold=0.5)

    # Random Forest
    rf_model = train_rf(X_train, y_train)
    evaluate_rf(rf_model, X_test, y_test, threshold=0.5)

    # ANN
    ann_model = train_ann(X_train, y_train, epochs=25, batch_size=32)
    evaluate_ann(ann_model, X_test, y_test, threshold=0.5)

    print("\n ALL MODELS TRAINED + EVALUATED SUCCESSFULLY\n")

    # Step 6 - Activation Blind Testing
    print("\n\n\n ACTIVATION BLIND TEST \n")
    run_activation_inference()


if __name__ == "__main__":
    main()
