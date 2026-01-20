# Employee Attrition Prediction Pipeline (AIBAS Project)

This project builds an end-to-end Machine Learning pipeline for predicting employee attrition (Attrition = Yes/No) using the IBM HR Analytics Employee Attrition dataset.

The pipeline performs:
- dataset download from GitHub
- preprocessing (target mapping + automatic useless feature removal)
- train/test split + activation (blind test) row creation
- preprocessing for ML (StandardScaler + OneHotEncoder)
- training of 4 models:
  - OLS Baseline
  - Logistic Regression
  - Random Forest
  - ANN (Neural Network)
- evaluation + plots (ROC, PR, Confusion Matrix)
- activation inference (blind test prediction + early-risk rule layer)

---

## Dataset

Dataset used:
**WA_Fn-UseC_-HR-Employee-Attrition.csv**

Stored automatically in your local machine

Downloaded from this repository (raw GitHub link is used internally): [Dataset](https://github.com/Adityaa4187/Aibas-Project)


---

## Project Structure
(<img width="380" height="576" alt="image" src="https://github.com/user-attachments/assets/89f19edf-3780-44b3-be6e-3dc0aaf5f422" />)

###  After running the pipeline, these folders/files are created automatically:

(<img width="450" height="717" alt="image-1" src="https://github.com/user-attachments/assets/b99cc958-fa7a-4713-b675-370a2e22602c" />)

## Install dependencies
### install using setup.py
    
    pip install -e .


## How to Run the Full Pipeline
### Simply run:
    
    python main.py


### This executes the complete workflow:

1. Download dataset from GitHub

2. Step-1 preprocessing + plots

3. Split into train/test + activation

4. Encode data (scaler + one-hot) + save preprocessor

5. Train and evaluate models + save plots

6. Run blind activation inference + save report


### Notes

1. The activation dataset is a single random row sampled from the test set (blind evaluation).

2. Automatic feature dropping:
    
    a. numeric low correlation with Attrition

    b. categorical low association with Attrition (Cramér’s V)

3. ANN uses a simple stable architecture for coursework submissions.

4. OLS is included only as a baseline.


Authors
1. Aditya Aiya

2. Meghana GN
