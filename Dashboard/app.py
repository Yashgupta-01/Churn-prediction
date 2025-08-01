import streamlit as st
import pandas as pd
import joblib

# Load model and preprocessor
preprocessor = joblib.load('/content/drive/MyDrive/Telecom churn prediction/models/preprocessor.joblib')
classifier = joblib.load('/content/drive/MyDrive/Telecom churn prediction/models/random_forest_classifier.pkl')

# List of expected columns
expected_columns = list(preprocessor.feature_names_in_)

# UI
st.title("📞 Telecom Churn Prediction Dashboard")

with st.form("form"):
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    pincode = st.text_input("Pincode", "110001")
    date_of_registration = st.text_input("Date of Registration", "2023-01-01")  # Optional use
    num_dependents = st.number_input("Number of Dependents", min_value=0, value=2)
    estimated_salary = st.number_input("Estimated Salary", min_value=0, value=50000)
    calls_made = st.number_input("Calls Made", min_value=0, value=100)
    sms_sent = st.number_input("SMS Sent", min_value=0, value=50)
    data_used = st.number_input("Data Used (in GB)", min_value=0.0, value=1.5)

    telecom_partner = st.selectbox("Telecom Partner", ['BSNL', 'Reliance Jio', 'Vodafone'])
    gender = st.selectbox("Gender", ['M', 'F'])
    state = st.selectbox("State", [
        'Arunachal Pradesh', 'Assam', 'Bihar', 'Chhattisgarh', 'Goa', 'Gujarat', 'Haryana',
        'Himachal Pradesh', 'Jharkhand', 'Karnataka', 'Kerala', 'Madhya Pradesh', 'Maharashtra',
        'Manipur', 'Meghalaya', 'Mizoram', 'Nagaland', 'Odisha', 'Punjab', 'Rajasthan', 'Sikkim',
        'Tamil Nadu', 'Telangana', 'Tripura', 'Uttar Pradesh', 'Uttarakhand', 'West Bengal'
    ])
    city = st.selectbox("City", ['Chennai', 'Delhi', 'Hyderabad', 'Kolkata', 'Mumbai'])

    submit = st.form_submit_button("Predict")

if submit:
    # Build base input dict
    input_data = {
        'age': age,
        'pincode': pincode,
        'date_of_registration': date_of_registration,
        'num_dependents': num_dependents,
        'estimated_salary': estimated_salary,
        'calls_made': calls_made,
        'sms_sent': sms_sent,
        'data_used': data_used,
    }

    # One-hot encoding manually for binary/categorical columns
    for col in expected_columns:
        if col.startswith('telecom_partner_'):
            input_data[col] = 1 if telecom_partner in col else 0
        elif col.startswith('gender_'):
            input_data[col] = 1 if gender in col else 0
        elif col.startswith('state_'):
            input_data[col] = 1 if state in col else 0
        elif col.startswith('city_'):
            input_data[col] = 1 if city in col else 0

    # Ensure all expected columns are present
    for col in expected_columns:
        if col not in input_data:
            input_data[col] = 0

    input_df = pd.DataFrame([input_data])

    try:
        processed = preprocessor.transform(input_df)
        prediction = classifier.predict(processed)
        prob = classifier.predict_proba(processed)[0][prediction[0]]

        st.subheader("Prediction Result:")
        st.success(f"{'Customer will churn' if prediction[0]==1 else 'Customer will not churn'}")
        st.info(f"Confidence: {prob:.2f}")
    except Exception as e:
        st.error(f"Error during prediction: {e}")
