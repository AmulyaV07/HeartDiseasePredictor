
import streamlit as st
import pickle
import pandas as pd

st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️")

# Load saved model
with open("knn_model.pkl", "rb") as f:
    data = pickle.load(f)

model = data["model"]
scaler = data["scaler"]
columns = data["columns"]

st.title("❤️ Heart Disease Prediction")
st.write("Enter the patient details below.")

# Input fields
age = st.number_input("Age", 1, 120, 40)
sex = st.selectbox("Sex", ["M", "F"])
chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "ASY", "TA"])
resting_bp = st.number_input("Resting Blood Pressure", value=120)
cholesterol = st.number_input("Cholesterol", value=200)
fasting_bs = st.selectbox("Fasting Blood Sugar", [0, 1])
resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
max_hr = st.number_input("Maximum Heart Rate", value=150)
exercise_angina = st.selectbox("Exercise Angina", ["N", "Y"])
oldpeak = st.number_input("Oldpeak", value=0.0)
st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

if st.button("Predict"):

    patient = pd.DataFrame([{
        "Age": age,
        "Sex": sex,
        "ChestPainType": chest_pain,
        "RestingBP": resting_bp,
        "Cholesterol": cholesterol,
        "FastingBS": fasting_bs,
        "RestingECG": resting_ecg,
        "MaxHR": max_hr,
        "ExerciseAngina": exercise_angina,
        "Oldpeak": oldpeak,
        "ST_Slope": st_slope
    }])

    # Apply the same one-hot encoding
    patient = pd.get_dummies(patient, drop_first=True, dtype=int)
    patient = patient.reindex(columns=columns, fill_value=0)

    # Scale using the saved scaler
    patient_scaled = scaler.transform(patient)

    # Predict
    prediction = model.predict(patient_scaled)[0]

    if prediction == 1:
        st.error("The model predicts the positive heart-disease class.")
    else:
        st.success("The model predicts the negative heart-disease class.")

    st.caption("Educational demonstration only—not a medical diagnosis.")
