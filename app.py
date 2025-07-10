import streamlit as st
import numpy as np
import pandas as pd
import torch
from hybridModel import HybridModel
import joblib

# --- Load model ---

model = HybridModel(31)
model.load_all(31, prefix="heart_model_")  

# --- Streamlit UI ---
st.title("💓 Heart Attack Risk Predictor")

user_data = {}

# --- Demographics ---
with st.expander("🧍 Demographics & Lifestyle"):
    for col in ["Sex", "AgeCategory", "GeneralHealth", "SmokerStatus", "ECigaretteUsage", "TetanusLast10Tdap"]:
        user_data[col] = st.selectbox(f"{col}:", ["0", "1"], key=col) 

# --- Physical Measures ---
with st.expander("📐 Physical Measurements"):
    for col in ["HeightInMeters", "WeightInKilograms", "BMI"]:
        user_data[col] = st.number_input(f"{col}:", min_value=0.0, step=0.1, key=col)

# --- Medical History ---
with st.expander("🩺 Medical Conditions"):
    for col in ["HadAngina", "HadStroke", "HadAsthma", "HadSkinCancer", "HadCOPD",
                "HadDepressiveDisorder", "HadKidneyDisease", "HadArthritis", "HadDiabetes"]:
        user_data[col] = st.selectbox(f"{col}:", ["No", "Yes"], key=col)

# --- Risk Factors ---
with st.expander("💉 Risk Factors"):
    for col in ["AlcoholDrinkers", "HIVTesting", "FluVaxLast12", "PneumoVaxEver",
                "HighRiskLastYear", "CovidPos"]:
        user_data[col] = st.selectbox(f"{col}:", ["No", "Yes"], key=col)

# --- Additional ---
with st.expander("🧪 Additional"):
    for col in ["DeafOrHardOfHearing", "BlindOrVisionDifficulty", "DifficultyConcentrating", "DifficultyWalking",
                "DifficultyDressingBathing", "DifficultyErrands", "ChestScan"]:
        user_data[col] = st.selectbox(f"{col}:", ["No", "Yes"], key=col)

# --- Predict ---
# Predict
if st.button("Predict"):
    df_input = pd.DataFrame([user_data])

    # Convert Yes/No to 1/0
    for col in df_input.columns:
        if df_input[col].iloc[0] == "Yes":
            df_input[col] = 1
        elif df_input[col].iloc[0] == "No":
            df_input[col] = 0
        else:
            df_input[col] = pd.to_numeric(df_input[col], errors="coerce")

    df_input.fillna(0, inplace=True)

    X_tensor = torch.tensor(df_input.values, dtype=torch.float32)

    # ✅ Use predict_proba from HybridModel
    proba, _ = model.predict_proba(X_tensor.numpy())

    st.markdown(f"### 🧪 Predicted Heart Attack Risk: `{proba[0]:.2%}`")
    if proba[0] > 0.5:
        st.error("⚠️ High Risk")
    else:
        st.success("✅ Low Risk")
