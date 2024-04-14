import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load scaler and SVM model
scaler = joblib.load('scaler.joblib')
svm_model = joblib.load('svm_model.joblib')

# Title of the Streamlit app
st.title('Predict Wine Quality')

# Inputs from the user
fixed_acidity = st.number_input("Fixed Acidity", min_value=0.0, max_value=100.0, step=0.1, value=7.0)
volatile_acidity = st.number_input("Volatile Acidity", min_value=0.0, max_value=10.0, step=0.01, value=0.5)
citric_acid = st.number_input("Citric Acid", min_value=0.0, max_value=10.0, step=0.01, value=0.5)
residual_sugar = st.number_input("Residual Sugar", min_value=0.0, max_value=100.0, step=0.1, value=2.0)
chlorides = st.number_input("Chlorides", min_value=0.0, max_value=10.0, step=0.01, value=0.08)
free_sulfur_dioxide = st.number_input("Free Sulfur Dioxide", min_value=0, max_value=500, step=1, value=30)
total_sulfur_dioxide = st.number_input("Total Sulfur Dioxide", min_value=0, max_value=1000, step=1, value=100)
density = st.number_input("Density", min_value=0.0, max_value=2.0, step=0.001, value=0.99)
pH = st.number_input("pH", min_value=0.0, max_value=14.0, step=0.01, value=3.5)
sulphates = st.number_input("Sulphates", min_value=0.0, max_value=10.0, step=0.01, value=0.6)
alcohol = st.number_input("Alcohol", min_value=0.0, max_value=100.0, step=0.1, value=10.0)

# Predict button
if st.button('Predict'):
    # Scale the input features
    input_features = scaler.transform([[fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides,
                                         free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates, alcohol]])
    # Make prediction
    prediction = svm_model.predict(input_features)

    # Display the prediction
    if prediction[0] == 1:
        st.success('Good quality wine.')
    else:
        st.error('Bad quality wine.')
