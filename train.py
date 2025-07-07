import pandas as pd
from sklearn.metrics import (f1_score, precision_score, recall_score, roc_auc_score,
                             confusion_matrix, ConfusionMatrixDisplay, precision_recall_curve)
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from hybridModel import HybridModel

def preprocess(df, encoders=None, scaler=None, fit=False):
    df = df.drop(columns=["PatientID"], errors="ignore")
    X = df.drop(columns=["HadHeartAttack"], errors="ignore")
    y = df["HadHeartAttack"] if "HadHeartAttack" in df else None

    if fit:
        encoders = {col: LabelEncoder().fit(X[col].astype(str)) for col in X.select_dtypes(include="object")}
        X = X.copy()
        for col, le in encoders.items():
            X[col] = le.transform(X[col].astype(str))
        scaler = StandardScaler().fit(X)
    else:
        for col, le in encoders.items():
            if col in X.columns:
                X[col] = le.transform(X[col].astype(str))
        if scaler is None:
            raise ValueError("Scaler must be provided when fit=False")

        # ✅ Ensure correct features
        expected_features = scaler.feature_names_in_
        for col in expected_features:
            if col not in X.columns:
                X[col] = 0
        X = X[expected_features]

    X_scaled = scaler.transform(X)
    return X_scaled, y, encoders, scaler


def find_best_threshold(y_test, preds):
    precisions, recalls, thresholds = precision_recall_curve(y_test, preds)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)  # Avoid division by zero
    best_idx = f1_scores.argmax()
    best_threshold = thresholds[best_idx]
    plt.plot(thresholds, f1_scores[:-1])
    plt.xlabel("Threshold")
    plt.ylabel("F1 Score")
    plt.title("F1 Score vs Threshold")
    plt.grid(True)
    plt.show()
    return best_threshold

def evaluate(y_true, y_prob, threshold=0.5):
    preds = (y_prob >= threshold).astype(int)
    metrics = {
        "Threshold": threshold,
        "Precision": precision_score(y_true, preds),
        "Recall": recall_score(y_true, preds),
        "F1 Score": f1_score(y_true, preds),
        "AUC-ROC": roc_auc_score(y_true, y_prob),
        "Accuracy": (preds == y_true).mean()
    }
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}" if key != "Threshold" else f"{key}: {value:.3f}")
    cm = confusion_matrix(y_true, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues", values_format=".0f")
    plt.show()

if __name__ == "__main__":
    train_df = pd.read_csv("p2.csv")
    test_df = pd.read_csv("p1.csv")
    
    X_train, y_train, enc, scl = preprocess(train_df, fit=True)
    X_test, y_test, _, _ = preprocess(test_df, encoders=enc, scaler=scl, fit=False)
    
    model = HybridModel(input_size=X_train.shape[1])
    model.encoders = enc
    model.scaler = scl
    model.train(X_train, y_train)
    
    y_prob, _ = model.predict_proba(X_test)
    threshold = find_best_threshold(y_test, y_prob)
    evaluate(y_test, y_prob, threshold)
    
    model.save_all("heart_model_")