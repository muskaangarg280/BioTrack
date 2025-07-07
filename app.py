import streamlit as st
import numpy as np
import pandas as pd
import torch
import joblib
from hybridModel import HybridModel

# --- Load model and scaler ---
scaler = joblib.load("heart_model_scaler.pkl")
input_size = scaler.mean_.shape[0]

model = HybridModel()
model.load_all(prefix="heart_model_", input_size=input_size)

encoders = model.encoders
feature_order = scaler.feature_names_in_

# --- Streamlit UI ---
st.title("💓 Heart Attack Risk Predictor")

user_data = {}

# --- Demographics ---
with st.expander("🧍 Demographics & Lifestyle"):
    for col in ["Sex", "AgeCategory", "GeneralHealth", "SmokerStatus", "ECigaretteUsage", "TetanusLast10Tdap"]:
        user_data[col] = st.selectbox(f"{col}:", encoders[col].classes_, key=col)

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

# --- Catch other fields if any ---
with st.expander("🧪 Additional"):
    for col in ["DeafOrHardOfHearing", "BlindOrVisionDifficulty", "DifficultyConcentrating", "DifficultyWalking",
                "DifficultyDressingBathing", "DifficultyErrands","ChestScan"]:
        user_data[col] = st.selectbox(f"{col}:", ["No", "Yes"], key=col)

# --- Predict ---
if st.button("Predict"):
    df_input = pd.DataFrame([user_data])

    # Step 1: Label encode multi-class categorical fields
    for col, le in encoders.items():
        if col in df_input.columns:
            val = df_input[col].astype(str)
            if val.iloc[0] not in le.classes_:
                st.error(f"Invalid input for {col}: {val.iloc[0]}")
                st.stop()
            df_input[col] = le.transform(val)

    # Step 2: Convert Yes/No to 1/0 for binary fields
    binary_fields = [  # These aren't label encoded, so map manually
        "HadAngina", "HadStroke", "HadAsthma", "HadSkinCancer", "HadCOPD",
        "HadDepressiveDisorder", "HadKidneyDisease", "HadArthritis", "HadDiabetes",
        "AlcoholDrinkers", "HIVTesting", "FluVaxLast12", "PneumoVaxEver",
        "HighRiskLastYear", "CovidPos", "DeafOrHardOfHearing", "BlindOrVisionDifficulty",
        "DifficultyConcentrating", "DifficultyWalking", "DifficultyDressingBathing",
        "DifficultyErrands", "ChestScan"
    ]
    for col in binary_fields:
        if col in df_input.columns:
            df_input[col] = df_input[col].map({"Yes": 1, "No": 0})


    # Step 3: Fill missing columns and reorder
    for col in feature_order:
        if col not in df_input.columns:
            df_input[col] = 0  # or another default
    df_input = df_input[feature_order]

    df_input.fillna(0, inplace=True)

    # Step 4: Scale the data
    X_scaled = scaler.transform(df_input)

    # Step 5: Predict
    try:
        proba, _ = model.predict_proba(X_scaled)
        final_prob = float(proba[0]) if not isinstance(proba, torch.Tensor) else proba.detach().cpu().numpy()[0]
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

    st.markdown(f"### 🧪 Predicted Heart Attack Risk: `{final_prob:.2%}`")
    if final_prob > 0.5:
        st.error("⚠️ High Risk")
    else:
        st.success("✅ Low Risk")
