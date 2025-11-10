import streamlit as st
import pandas as pd
import numpy as np
import pickle

# -------------------------------
# Load the trained model
# -------------------------------
model = pickle.load(open("gradient_boosting_classifier.pkl", "rb"))

# -------------------------------
# App title and description
# -------------------------------
st.set_page_config(page_title="E-commerce Fraud Detection", layout="wide")

st.title("🛒 E-commerce Fraud Detection")
st.markdown("""
This app uses a trained **Gradient Boosting (AdaBoost-style)** model to predict fraudulent transactions.
""")

# -------------------------------
# Input features
# -------------------------------
st.header("🔢 Enter Transaction Details")

col1, col2, col3 = st.columns(3)

with col1:
    amount = st.number_input("Transaction Amount ($)", min_value=0.0, max_value=10000.0, value=100.0)
    time = st.number_input("Transaction Time (in seconds)", min_value=0.0, max_value=86400.0, value=3600.0)

with col2:
    age = st.number_input("Customer Age", min_value=18, max_value=90, value=30)
    location_match = st.selectbox("Shipping & Billing Address Match?", ["Yes", "No"])

with col3:
    device_type = st.selectbox("Device Type", ["Mobile", "Desktop", "Tablet"])
    previous_fraud = st.selectbox("Previous Fraud History?", ["No", "Yes"])

# -------------------------------
# Data preprocessing (example)
# -------------------------------
input_data = pd.DataFrame({
    'amount': [amount],
    'time': [time],
    'age': [age],
    'location_match': [1 if location_match == "Yes" else 0],
    'device_type': [0 if device_type == "Mobile" else (1 if device_type == "Desktop" else 2)],
    'previous_fraud': [1 if previous_fraud == "Yes" else 0]
})

# -------------------------------
# Prediction
# -------------------------------
if st.button("🚨 Detect Fraud"):
    prediction = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0][1] * 100

    if prediction == 1:
        st.error(f"⚠️ Fraudulent Transaction Detected! (Confidence: {proba:.2f}%)")
    else:
        st.success(f"✅ Legitimate Transaction (Confidence: {100 - proba:.2f}%)")

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.markdown("💡 *Model: Gradient Boosting Classifier (AdaBoost style)*")
