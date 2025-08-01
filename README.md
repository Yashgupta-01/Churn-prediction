# 📞 Telecom Churn Prediction Dashboard

This project predicts customer churn in a telecom company using machine learning and provides an interactive web-based dashboard for real-time predictions.

## 🚀 Project Overview

Churn prediction is critical for telecom businesses to identify customers likely to leave. This system allows users to input customer data and receive churn predictions with confidence scores using a trained machine learning model.

The project is built with:
- Python 🐍
- Scikit-learn for ML modeling
- Streamlit for dashboard UI
- Ngrok for exposing the app (in Colab)
- Joblib for saving/loading models

## 🧠 Machine Learning Pipeline

1. **Data Cleaning & Preprocessing**  
   - Handled missing values, encoded categorical variables.
   - Used `StandardScaler` and `OneHotEncoder` via `ColumnTransformer`.

2. **Model Training**
   - Trained a `RandomForestClassifier` with `max_depth=15`, `n_estimators=100`.
   - Pipeline built with `sklearn.pipeline.Pipeline`.

3. **Model Export**
   - Trained model and preprocessing pipeline saved as:
     - `random_forest_classifier.pkl`
     - `preprocessor.joblib`

## 🖥️ Dashboard (Streamlit App)

- Inputs:
  - Age, Salary, Calls Made, SMS Sent, Data Used, Telecom Partner, Gender, City
- Output:
  - Churn Prediction (`Churned` or `Not Churned`)
  - Confidence score
- Accuracy: 79.94%
