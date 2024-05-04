import streamlit as st
import joblib

# Load scaler and SVM model
scaler = joblib.load('scaler.joblib')
svm_model = joblib.load('voting_model.joblib')

# Title of the Streamlit app
st.title('Predict Wine Quality')

# Inputs from the user
volatile_acidity = st.number_input("Volatile Acidity", min_value=0.0, max_value=10.0, step=0.01, value=0.5)
citric_acid = st.number_input("Citric Acid", min_value=0.0, max_value=10.0, step=0.01, value=0.5)
sulphates = st.number_input("Sulphates", min_value=0.0, max_value=10.0, step=0.01, value=0.6)
alcohol = st.number_input("Alcohol", min_value=0.0, max_value=100.0, step=0.1, value=10.0)

# Predict button
if st.button('Predict Quality'):
    # Scale the input features
    input_features = scaler.transform([[alcohol,sulphates, citric_acid,volatile_acidity]])
    # Make prediction
    prediction = svm_model.predict(input_features)

    # Display the prediction
    if prediction[0] == 1:
        st.success('Good quality wine.')
    else:
        st.error('Bad quality wine.')
