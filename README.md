# 📉 Telecom Churn Prediction

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.25+-red.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3+-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 📌 Problem Statement

In the highly competitive telecom industry, Customer Acquisition Cost (CAC) is a significant expense—it costs 5-7x more to acquire a new customer than to retain an existing one. Unpredictable customer churn leads to massive revenue leakage because companies usually only realize a customer is unhappy *after* they have left for a competitor.

## 🚀 Solution

This project provides an end-to-end Machine Learning solution to proactively predict customer churn. By analyzing demographic data, location, and telecom usage patterns, the model predicts the likelihood of a customer leaving. 

This enables Customer Success and Retention teams to transition from a **reactive** approach to a **proactive** strategy, offering targeted retention discounts to high-risk customers before they churn.

## 💡 Key Features

- **End-to-End ML Pipeline:** Seamless data preprocessing (StandardScaler & OneHotEncoder) bundled with the model into a single robust Scikit-Learn Pipeline.
- **Predictive Accuracy:** Achieved ~80% accuracy using a tuned Random Forest Classifier, outperforming the baseline Logistic Regression model.
- **Real-Time Interactive Dashboard:** A premium, user-friendly web application built with Streamlit for frontline employees to assess customer risk instantly.
- **Probabilistic Risk Alerts:** Outputs the exact probability percentage of churn (e.g., 85% High Risk) rather than just a boolean value, allowing businesses to prioritize critical accounts.

## 🛠️ Technology Stack

- **Data Processing & ML:** `pandas`, `numpy`, `scikit-learn`
- **Model Serialization:** `joblib`
- **Web Dashboard:** `streamlit`

## ⚙️ Installation & Local Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Yashgupta-01/Churn-prediction.git
   cd Churn-prediction
   ```

2. **Create a virtual environment (Optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. **Install the dependencies:**
   ```bash
   pip install -r dashboard/requirements.txt
   ```

## 🖥️ Usage

**To run the interactive Streamlit dashboard:**
```bash
streamlit run dashboard/app.py
```
The dashboard will open automatically in your default web browser at `http://localhost:8501`. 

*(Simply enter the customer's demographics and telecom usage to instantly see their churn probability!)*

## 🧠 Model Training

If you wish to retrain the machine learning pipeline from scratch using the raw data:
```bash
python scripts/train_pipeline.py
```
This script will read the raw dataset, build the ColumnTransformer and RandomForest model, split the data, train the pipeline, and save the updated `.pkl` artifact directly into the `models/` directory.

---
*Developed by [Yash Gupta](https://github.com/Yashgupta-01).*
