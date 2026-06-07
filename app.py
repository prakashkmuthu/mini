import streamlit as st
import requests

st.title("Student Pass/Fail Prediction")

study_hours = st.number_input("Study Hours", min_value=0.0)
attendance = st.number_input("Attendance", min_value=0.0, max_value=100.0)
previous_score = st.number_input("Previous Score", min_value=0.0, max_value=100.0)
assignment_score = st.number_input("Assignment Score", min_value=0.0, max_value=100.0)
sleep_hours = st.number_input("Sleep Hours", min_value=0.0)
internet_usage = st.number_input("Internet Usage", min_value=0.0)

if st.button("Predict"):
    data = {
        "study_hours": study_hours,
        "attendance": attendance,
        "previous_score": previous_score,
        "assignment_score": assignment_score,
        "sleep_hours": sleep_hours,
        "internet_usage": internet_usage
    }

    response = requests.post("http://127.0.0.1:8000/predict", json=data)

    if response.status_code == 200:
        result = response.json()
        st.success(f"Prediction: {result['result']}")
    else:
        st.error("Something went wrong")