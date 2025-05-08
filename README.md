---
title: Bankruptcy Prediction System
emoji: 📈
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
---

# Bankruptcy Prediction System

This Gradio-based web application predicts company bankruptcy using machine learning models trained on financial data. It uses 30 selected financial features and supports four models: Logistic Regression, Random Forest, XGBoost, and Neural Network (MLP).

## Features
- Input values for 30 financial features (e.g., ROA (C), Operating Gross Margin, etc.).
- Choose from four trained models to make predictions.
- View prediction results (Bankrupt or Non-bankrupt) and bankruptcy probability.
- Display model performance metrics (accuracy, precision, recall, F1-score).

## Setup
1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`.
3. Ensure `data_rename.csv` and model `.pkl` files are in the root directory.
4. Run the app: `python app.py`.

## Dependencies
- Python 3.8+
- Gradio
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Imbalanced-learn
- Matplotlib
- Seaborn
- Joblib
- Openpyxl

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
