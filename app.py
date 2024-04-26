import streamlit as st
import joblib

# Load scaler and voting ensemble model
scaler = joblib.load('minmax_scaler.joblib')
ensemble_model = joblib.load('voting_ensemble_model.joblib')

# Title of the Streamlit app
st.title('Predict Anomalies')

# Inputs from the user
logged_in = st.selectbox("Logged In", [0, 1], format_func=lambda x: '0' if x == 0 else '1')
count = st.number_input("Count", min_value=0, step=1, value=50)
dst_host_count = st.number_input("Destination Host Count", min_value=0, step=1, value=50)
protocol_type = st.selectbox("Protocol Type", [0, 1,2])
srv_count = st.number_input("Service Count", min_value=0, step=1, value=50)
dst_host_diff_srv_rate = st.number_input("Destination Host Different Service Rate", min_value=0.0, step=0.01, value=0.5)
dst_host_same_src_port_rate = st.number_input("Destination Host Same Source Port Rate", min_value=0.0, step=0.01, value=0.5)
dst_host_srv_diff_host_rate = st.number_input("Destination Host Server Different Host Rate", min_value=0.0, step=0.01, value=0.5)
hot = st.number_input("Hot", min_value=0, step=1, value=0)
root_shell = st.number_input("Root Shell", min_value=0, step=1, value=0)

# Predict button
if st.button('Predict'):
    # Scale the input features
    input_features = scaler.transform([[logged_in, count, dst_host_count, protocol_type, srv_count, dst_host_diff_srv_rate, dst_host_same_src_port_rate, dst_host_srv_diff_host_rate, hot, root_shell]])

    # Make prediction
    prediction = ensemble_model.predict(input_features)

    # Display the prediction
    st.success(prediction[0])
