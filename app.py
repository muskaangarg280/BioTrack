import streamlit as st
import numpy as np
import pandas as pd
import torch
from hybridModel import HybridModel

# --- Load model ---
model = HybridModel(31)
model.load_all(31, prefix="heart_model_")  

# --- App Title ---
st.title("Heart Attack Risk Assessment Tool")
st.markdown("""
This tool uses a hybrid machine learning model to predict an individual's risk of experiencing a heart attack based on health, lifestyle, and medical history information.
""")

user_data = {}

# --- Section: Demographics & Lifestyle ---
st.header("Demographics & Lifestyle")
with st.container():
    user_data["Sex"] = st.selectbox("Sex", ["Male", "Female"])
    user_data["AgeCategory"] = st.selectbox("Age Category", [
        "18-24", "25-29", "30-34", "35-39", "40-44", "45-49",
        "50-54", "55-59", "60-64", "65-69", "70-74", "75-79", "80+"
    ])
    user_data["GeneralHealth"] = st.selectbox("General Health", [
        "Excellent", "Very good", "Good", "Fair", "Poor"
    ])
    user_data["SmokerStatus"] = st.selectbox("Smoker Status", [
        "Never smoked", "Former smoker", "Current smoker"
    ])
    user_data["ECigaretteUsage"] = st.selectbox("E-Cigarette Usage", [
        "Never used", "Tried once or twice", "Occasional user", "Regular user"
    ])
    user_data["TetanusLast10Tdap"] = st.selectbox("Tetanus Vaccine (Last 10 Years)", [
        "Yes", "No", "Not sure"
    ])

# --- Section: Physical Measurements ---
st.header("Physical Measurements")
with st.container():
    for col in ["HeightInMeters", "WeightInKilograms", "BMI"]:
        user_data[col] = st.number_input(f"{col.replace('In', ' (in ').replace('Kilograms', 'kg)').replace('Meters', 'm)')}", min_value=0.0, step=0.1)

# --- Section: Medical History ---
st.header("Medical History")
with st.container():
    for col in ["HadAngina", "HadStroke", "HadAsthma", "HadSkinCancer", "HadCOPD",
                "HadDepressiveDisorder", "HadKidneyDisease", "HadArthritis", "HadDiabetes"]:
        user_data[col] = st.selectbox(col.replace("Had", "History of "), ["No", "Yes"])

# --- Section: Risk Factors ---
st.header("Other Health Factors")
with st.container():
    for col in ["AlcoholDrinkers", "HIVTesting", "FluVaxLast12", "PneumoVaxEver",
                "HighRiskLastYear", "CovidPos"]:
        user_data[col] = st.selectbox(col.replace("Vax", " Vaccine").replace("CovidPos", "Tested Positive for COVID-19"), ["No", "Yes"])

# --- Section: Accessibility & Functional Limitations ---
st.header("Additional Information")
with st.container():
    for col in ["DeafOrHardOfHearing", "BlindOrVisionDifficulty", "DifficultyConcentrating", "DifficultyWalking",
                "DifficultyDressingBathing", "DifficultyErrands", "ChestScan"]:
        user_data[col] = st.selectbox(col.replace("Difficulty", "Issues with ").replace("Or", " or "), ["No", "Yes"])

# --- Predict Button ---
if st.button("Assess Risk", key="predict_button"):
    df_input = pd.DataFrame([user_data])

    # Convert Yes/No to 1/0
    for col in df_input.columns:
        if df_input[col].iloc[0] == "Yes":
            df_input[col] = 1
        elif df_input[col].iloc[0] == "No":
            df_input[col] = 0

    # Apply encoders
    for col, le in model.encoders.items():
        if col in df_input.columns:
            df_input[col] = df_input[col].astype(str).apply(
                lambda x: x if x in le.classes_ else le.classes_[0]
            )
            df_input[col] = le.transform(df_input[col])

    # Ensure feature set matches training
    expected_cols = model.scaler.feature_names_in_
    for col in expected_cols:
        if col not in df_input.columns:
            df_input[col] = 0
    df_input = df_input[expected_cols]

    # Scale and predict
    X_scaled = model.scaler.transform(df_input)
    proba, _ = model.predict_proba(X_scaled)

    st.subheader("Prediction Result")
    st.write(f"Estimated Risk: **{proba[0]:.2%}**")
    if proba[0] > 0.5:
        st.error("This individual is at high risk for a heart attack.")
    else:
        st.success("This individual is at low risk for a heart attack.")
