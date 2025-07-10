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
📂 BioTrackAI/
├── app.py              # Streamlit app for user interaction
├── train.py            # End-to-end training pipeline
├── DNNmodel.py         # Deep Neural Network architecture (PyTorch)
├── hybridModel.py      # Hybrid ensemble with DNN, RF, and meta model
├── p1.csv              # Testing dataset
├── p2.csv              # Training dataset
├── requirements.txt    # Python dependencies
├── requirements.txt
├── requirements.txt
├── requirements.txt
├── requirements.txt
├── requirements.txt 
└── README.md           # This file
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
