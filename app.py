import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="centered"
)

st.title("🎓 Student Performance Predictor")
st.write(
    "Predict whether a student is likely to pass "
    "based on study habits and previous grades."
)

# Load the trained machine-learning model
saved_data = joblib.load("student_model.pkl")
model = saved_data["model"]
features = saved_data["features"]

st.subheader("Enter Student Details")

with st.form("student_form"):
    studytime = st.selectbox(
        "Weekly study time",
        options=[1, 2, 3, 4],
        format_func=lambda x: {
            1: "Less than 2 hours",
            2: "2–5 hours",
            3: "5–10 hours",
            4: "More than 10 hours"
        }[x]
    )

    absences = st.number_input(
        "Number of absences",
        min_value=0,
        max_value=100,
        value=0,
        step=1
    )

    G1 = st.slider("First period grade (0–20)", 0, 20, 10)
    G2 = st.slider("Second period grade (0–20)", 0, 20, 10)

    failures = st.selectbox(
        "Number of previous class failures",
        [0, 1, 2, 3]
    )

    submitted = st.form_submit_button("Predict Performance")

if submitted:
    student_data = pd.DataFrame(
        [[studytime, absences, G1, G2, failures]],
        columns=["studytime", "absences", "G1", "G2", "failures"]
    )

    prediction = model.predict(student_data)[0]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.success("The model predicts that the student is likely to pass.")
    else:
        st.warning("The model predicts that the student may be at risk of failing.")

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(student_data)[0]
        st.write(f"Estimated pass probability: {probabilities[1] * 100:.1f}%")

    st.caption(
        "This is an educational estimate based on the training dataset, "
        "not a guarantee of actual academic performance."
    )