# credit-scoring-model
Machine learning project for predicting creditworthiness from historical financial data.
# Credit Scoring Model

A complete machine learning project for predicting creditworthiness from historical financial data.

## Project Overview

This project develops an end-to-end machine learning pipeline for credit-risk classification using the Statlog German Credit dataset from the UCI Machine Learning Repository.

The system predicts whether an applicant belongs to:

- Good Credit
- Bad Credit

The project includes:

- Data loading
- Data exploration
- Data preprocessing
- Feature engineering
- Logistic Regression
- Random Forest
- XGBoost
- 5-fold cross-validation
- Hyperparameter tuning
- Model evaluation
- SHAP explainability
- Model persistence
- Streamlit prediction application

## Dataset

Dataset:

Statlog (German Credit Data)

Source:

https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data

The dataset contains 1,000 instances and 20 predictor attributes.

The original target is:

- 1 = Good credit
- 2 = Bad credit

For this project it is converted to:

- 0 = Good credit
- 1 = Bad credit

## Project Pipeline

```text
UCI Dataset
    ↓
Data Loading
    ↓
Data Validation
    ↓
Train/Test Split
    ↓
Feature Engineering
    ↓
Preprocessing
    ↓
Logistic Regression
    ↓
Random Forest
    ↓
XGBoost
    ↓
5-Fold Cross Validation
    ↓
Hyperparameter Tuning
    ↓
Final Evaluation
    ↓
SHAP Explainability
    ↓
Saved Model
    ↓
Streamlit Application
