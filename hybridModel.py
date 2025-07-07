import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import joblib
from DNNmodel import DNN
from sklearn.preprocessing import StandardScaler

class HybridModel:
    def __init__(self, input_size=None):
        self.dnn = DNN(input_size) if input_size else None
        self.rf = None
        self.meta_model = None
        self.encoders = None
        self.scaler = None
        self.meta_scaler = None

    def train(self, X_train, y_train):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dnn = DNN(input_size=X_train.shape[1]).to(device)
        
        pos_weight = torch.tensor((len(y_train) - y_train.sum()) / y_train.sum(), dtype=torch.float32).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.Adam(self.dnn.parameters(), lr=0.001)
        
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                torch.tensor(X_train, dtype=torch.float32).to(device),
                torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1).to(device)
            ), batch_size=256, shuffle=True
        )
        
        self.dnn.train()
        for epoch in range(15):
            total_loss = 0
            for xb, yb in loader:
                optimizer.zero_grad()
                logits = self.dnn(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch+1}/15 – Loss: {total_loss:.4f}")
        
        self.dnn.eval()
        with torch.no_grad():
            dnn_preds_train = torch.sigmoid(self.dnn(torch.tensor(X_train, dtype=torch.float32).to(device))).cpu().numpy().flatten()
        
        self.rf = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
        self.rf.fit(X_train, y_train)
        rf_preds_train = self.rf.predict_proba(X_train)[:, 1]
        
        stack_train = np.vstack([dnn_preds_train, rf_preds_train]).T
        self.meta_scaler = StandardScaler().fit(stack_train)
        stack_train_scaled = self.meta_scaler.transform(stack_train)
        
        self.meta_model = LogisticRegression(class_weight="balanced", max_iter=1000)
        self.meta_model.fit(stack_train_scaled, y_train)

    def predict_proba(self, X):
        with torch.no_grad():
            dnn_probs = torch.sigmoid(self.dnn(torch.tensor(X, dtype=torch.float32))).cpu().numpy().flatten()
        
        rf_probs = self.rf.predict_proba(X)[:, 1]
        dnn_probs = np.clip(dnn_probs, 1e-6, 1 - 1e-6)
        rf_probs = np.clip(rf_probs, 1e-6, 1 - 1e-6)
        
        stacked = np.vstack([dnn_probs, rf_probs]).T
        stacked = self.meta_scaler.transform(stacked)
        
        meta_probs = self.meta_model.predict_proba(stacked)[:, 1]
        return meta_probs, None

    def save_all(self, prefix=""):
        torch.save(self.dnn.state_dict(), f"{prefix}dnn_model.pt")
        joblib.dump(self.rf, f"{prefix}rf_model.pkl")
        joblib.dump(self.meta_model, f"{prefix}meta_model.pkl")
        joblib.dump(self.encoders, f"{prefix}encoders.pkl")
        joblib.dump(self.scaler, f"{prefix}scaler.pkl")
        joblib.dump(self.meta_scaler, f"{prefix}meta_scaler.pkl")

    def load_all(self, input_size, prefix=""):
        self.dnn = DNN(input_size)
        self.dnn.load_state_dict(torch.load(f"{prefix}dnn_model.pt", map_location="cpu"))
        self.dnn.eval()
        self.rf = joblib.load(f"{prefix}rf_model.pkl")
        self.meta_model = joblib.load(f"{prefix}meta_model.pkl")
        self.encoders = joblib.load(f"{prefix}encoders.pkl")
        self.scaler = joblib.load(f"{prefix}scaler.pkl")
        self.meta_scaler = joblib.load(f"{prefix}meta_scaler.pkl")