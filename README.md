# BioTrack: Heart Attack Risk Prediction System

[Launch App ](https://biotrackai-pjkruicdn7fcmfgkj4x6pe.streamlit.app/)

**BioTrackAI** is a hybrid machine learning system designed to predict the risk of heart attacks using both clinical and lifestyle data. It combines Random Forests, Deep Neural Networks, and Logistic Regression in a stacked ensemble for robust and explainable predictions.

---
## Features

- **Hybrid Architecture**:
  - Random Forest + Deep Neural Network
  - Logistic Regression meta-learner on stacked outputs
- **Strong Performance**:
  - Accuracy: ~94%
  - AUC-ROC: ~0.79
  - F1 Score optimized with a custom threshold search
- **Interactive UI**:
  - Built with Streamlit
  - Accepts medical, lifestyle, and demographic inputs
---
## Try It Yourself

Head to [biotrackai streamlit app](https://biotrackai-pjkruicdn7fcmfgkj4x6pe.streamlit.app/)  
Enter your demographic, medical, and lifestyle data to get an immediate risk score.

---

## Model Performance

- **Accuracy**: `94%` on held-out test data  
- **Metrics**:
  - Precision, Recall, F1 Score, AUC-ROC  
  - Confusion Matrix  
  - Predicted vs Actual Probability plots
 
## Tech Stack

| Layer         | Technology                  |
|---------------|-----------------------------|
| Modeling      | PyTorch, scikit-learn       |
| Preprocessing | pandas, numpy, sklearn      |
| Evaluation    | matplotlib, sklearn metrics |
| UI            | Streamlit                   |
| Packaging     | joblib, torch               |

---


## 📁 File Structure

```
├── app.py                         # Streamlit UI
├── train.py                       # Training script
├── DNNmodel.py                    # PyTorch DNN definition
├── hybridModel.py                 # Hybrid model wrapper - DNN + Random Classifier
├── p1.csv                         # Test data
├── p2.csv                         # Training data
├── requirements.txt               # Dependencies
├── .gitignore
│
├── heart_model_dnn_model.pt       # Trained DNN weights
├── heart_model_rf_model.pkl       # Random Forest model
├── heart_model_meta_model.pkl     # Meta model (LogReg)
├── heart_model_encoders.pkl       # LabelEncoders for categorical data
├── heart_model_scaler.pkl         # Scaler for input features
├── heart_model_meta_scaler.pkl    # Scaler for meta model input
└── README.md                      # This file
```

 ---

## Future Enhancements

- Add multi-language support
- Add SHAP analysis
- Make it mobile-responsive
- Exportable health reports
- Try alternative ML models (e.g. XGBoost, Random Forest)

---

 ## Dataset Attribution

This project uses the [Patients Data for Medical Field](https://www.kaggle.com/datasets/tarekmuhammed/patients-data-for-medical-field) dataset by **Tarek Muhammed** (Kaggle).  
The dataset is used under Kaggle’s [terms of use](https://www.kaggle.com/terms) for research and educational purposes.

---
