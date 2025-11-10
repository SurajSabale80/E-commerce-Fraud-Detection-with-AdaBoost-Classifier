import streamlit as st
import pandas as pd
import numpy as np
import pickle

# -------------------------------
# Load the trained AdaBoost model
# -------------------------------
model = pickle.load(open("ada_model.pkl", "rb"))

# -------------------------------
# App title and description
# -------------------------------
st.set_page_config(page_title="E-commerce Fraud Detection", layout="wide")

st.title("🛒 E-commerce Fraud Detection")
st.markdown("""
This app uses a trained **AdaBoost Classifier** to predict whether an e-commerce transaction is fraudulent.
""")

# -------------------------------
# Input features
# -------------------------------
st.header("🔢 Enter Transaction Details")

col1, col2, col3 = st.columns(3)

with col1:
    amount = st.number_input("Transaction Amount ($)", min_value=0.0, max_value=10000.0, value=100.0)
    time = st.number_input("Transaction Time (seconds since start of day)", min_value=0.0, max_value=86400.0, value=3600.0)

with col2:
    age = st.number_input("Customer Age", min_value=18, max_value=90, value=30)
    location_match = st.selectbox("Shipping & Billing Address Match?", ["Yes", "No"])

with col3:
    device_type = st.selectbox("Device Type", ["Mobile", "Desktop", "Tablet"])
    previous_fraud = st.selectbox("Previous Fraud History?", ["No", "Yes"])

# -------------------------------
# Data preprocessing (convert categorical to numeric)
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
# -------------------------------
# Prediction (with debug)
# -------------------------------
if st.button("🚨 Detect Fraud"):
    try:
        prediction = model.predict(input_data.values)[0]
        proba = model.predict_proba(input_data.values)[0][1] * 100

        if prediction == 1:
            st.error(f"⚠️ Fraudulent Transaction Detected! (Confidence: {proba:.2f}%)")
        else:
            st.success(f"✅ Legitimate Transaction (Confidence: {100 - proba:.2f}%)")

    except ValueError as e:
        st.error("❌ Input shape or features mismatch!")
        st.write("Error details:", str(e))

        if hasattr(model, 'n_features_in_'):
            st.write(f"Model expects {model.n_features_in_} features.")
            st.write(f"Input data shape: {input_data.shape}")
        if hasattr(model, 'feature_names_in_'):
            st.write("Expected feature names:", list(model.feature_names_in_))
        st.write("Input DataFrame columns:", list(input_data.columns))

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.markdown("💡 *Model: AdaBoost Classifier trained on e-commerce transaction data.*")

