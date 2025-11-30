import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.title("❤️ Heart Disease Prediction App")


MODEL_FILES = ["tree_heart.pkl", "Decisiontree_heart.pkl", "model.pkl"]
SCALER_FILES = ["Scaler.pkl", "scaler.pkl"]

model = None
scaler = None

# Load model
for file in MODEL_FILES:
    if os.path.exists(file):
        obj = joblib.load(file)
        # Accept only real ML models (objects that have .predict)
        if hasattr(obj, "predict"):
            model = obj
            st.success(f"Loaded model: {file}")
            break

if model is None:
    st.error("❌ No valid model found. Make sure 'tree_heart.pkl' or 'Decisiontree_heart.pkl' is in the same folder.")
    st.stop()

# Load scaler
for file in SCALER_FILES:
    if os.path.exists(file):
        obj = joblib.load(file)
        # Accept only real scalers (objects that have transform)
        if hasattr(obj, "transform"):
            scaler = obj
            st.success(f"Loaded scaler: {file}")
            break

if scaler is None:
    st.warning("⚠️ Scaler not found. Predictions will use raw values.")


age = st.slider("Age", 18, 100, 50)
sex = st.selectbox("Sex", ["M", "F"])
cp = st.selectbox("Chest Pain Type", ["typical_angina", "atypical_angina",
                                      "non_anginal_pain", "asymptomatic"])

trestbps = st.number_input("Resting BP (trestbps)", 80, 220, 120)
chol = st.number_input("Cholesterol (chol)", 100, 600, 200)

fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])
restecg = st.selectbox("Resting ECG", ["normal", "ST-T abnormality", "LVH"])
thalach = st.number_input("Max Heart Rate (thalach)", 60, 220, 150)
exang = st.selectbox("Exercise Induced Angina (exang)", ["N", "Y"])
oldpeak = st.slider("Oldpeak (ST Depression)", 0.0, 6.0, 1.0, 0.1)
slope = st.selectbox("ST Slope", ["up", "flat", "down"])
ca = st.selectbox("Number of Major Vessels (ca)", [0, 1, 2, 3, 4])
thal = st.selectbox("Thal", [0, 1, 2, 3])


sex_map = {"M": 1, "F": 0}
cp_map = {"typical_angina": 0, "atypical_angina": 1,
          "non_anginal_pain": 2, "asymptomatic": 3}
fbs_map = {"No": 0, "Yes": 1}
restecg_map = {"normal": 0, "ST-T abnormality": 1, "LVH": 2}
exang_map = {"N": 0, "Y": 1}
slope_map = {"up": 0, "flat": 1, "down": 2}


if st.button("Predict"):
    data = {
        "age": age,
        "sex": sex_map[sex],
        "cp": cp_map[cp],
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs_map[fbs],
        "restecg": restecg_map[restecg],
        "thalach": thalach,
        "exang": exang_map[exang],
        "oldpeak": oldpeak,
        "slope": slope_map[slope],
        "ca": ca,
        "thal": thal
    }

    df = pd.DataFrame([data])

    # Correct column order (13 features)
    expected_columns = [
        "age", "sex", "cp", "trestbps", "chol", "fbs",
        "restecg", "thalach", "exang", "oldpeak",
        "slope", "ca", "thal"
    ]
    df = df[expected_columns]

    # Scale safely
    if scaler is not None:
        try:
            df_scaled = scaler.transform(df)
        except:
            df_scaled = df
            st.warning("Scaler failed. Using raw values.")
    else:
        df_scaled = df

    # Predict
    try:
        pred = model.predict(df_scaled)[0]
        if pred == 1:
            st.error("⚠️ High risk of heart disease!")
        else:
            st.success("✔️ Low risk of heart disease!")
    except Exception as e:
        st.error(f"Model prediction failed: {e}")
