import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.svm import SVC
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import io
import base64

'''
The CancerPredictor class is a PyTorch neural network model designed for binary classification tasks.
we utilize a feedforward neural network architecture with two hidden (ReLU) and dropout layers.
0 represents Malignant (Cancerous) and 1 represents Benign (Not Cancerous).
'''
class CancerPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim=32):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1), nn.Sigmoid()
        )
    def forward(self, x): return self.network(x)


'''
The ModelManager class is responsible for managing the training, prediction, and visualization of a 
cancer prediction model. It provides methods to load and preprocess the data, train the model, make 
predictions, and generate plots to visualize the model's performance.
'''
class ModelManager:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.pca = None
        self.pca_svc = None
        self.is_trained = False
        self.feature_names = None
        self.target_names = None
        self.dataset_name = None
        self.X_train_pca, self.X_test_pca = None, None
        self.y_train, self.y_test = None, None
        self.X_train, self.X_test = None, None
        self.history = {"losses": [], "accuracies": []}

    def get_available_datasets(self):
        return ["breast_cancer", "synthetic_churn"]

    def load_data(self, dataset_name="breast_cancer"):
        self.dataset_name = dataset_name
        self.is_trained = False # Reset trained state
        self.history = {"losses": [], "accuracies": []}
        self.model = None
        self.pca_svc = None

        if dataset_name == "breast_cancer":
            data = load_breast_cancer()
            X, y = data.data, data.target
            self.feature_names = list(data.feature_names)
            self.target_names = ["Malignant", "Benign"]
        else: # synthetic_churn
            X, y = make_classification(
                n_samples=1000, n_features=15, n_informative=10, 
                n_redundant=2, random_state=42
            )
            self.feature_names = [f"Account_Feature_{i}" for i in range(15)]
            self.target_names = ["Retained", "Churned"]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(self.X_train)
        X_test_scaled = self.scaler.transform(self.X_test)
        
        self.pca = PCA(n_components=2)
        self.X_train_pca = self.pca.fit_transform(X_train_scaled)
        self.X_test_pca = self.pca.transform(X_test_scaled)
        
        return {"features": self.feature_names, "train_size": len(self.y_train), "target_names": self.target_names}

    def train(self, epochs=50, lr=0.01, hidden_dim=32):
        input_dim = self.X_train.shape[1] # DYNAMICALLY SIZES BASED ON NEW DATASET
        self.model = CancerPredictor(input_dim, hidden_dim)
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.BCELoss()
        
        X_train_tensor = torch.FloatTensor(self.scaler.transform(self.X_train))
        y_train_tensor = torch.FloatTensor(self.y_train).unsqueeze(1)
        loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=32, shuffle=True)
        
        self.history = {"losses": [], "accuracies": []}
        for _ in range(epochs):
            self.model.train()
            epoch_loss, correct, total = 0.0, 0, 0
            for X_b, y_b in loader:
                optimizer.zero_grad()
                out = self.model(X_b)
                loss = criterion(out, y_b)
                loss.backward(); optimizer.step()
                epoch_loss += loss.item()
                correct += ((out >= 0.5).float() == y_b).sum().item()
                total += y_b.size(0)
            self.history["losses"].append(epoch_loss / len(loader))
            self.history["accuracies"].append(correct / total)
            
        self.model.eval()
        with torch.no_grad():
            X_test_tensor = torch.FloatTensor(self.scaler.transform(self.X_test))
            preds = (self.model(X_test_tensor) >= 0.5).numpy().flatten()
        
        self.pca_svc = SVC(kernel='rbf', gamma=0.5)
        self.pca_svc.fit(self.X_train_pca, self.y_train)
        
        self.is_trained = True
        return float(accuracy_score(self.y_test, preds)), confusion_matrix(self.y_test, preds).tolist()

    def predict(self, features):
        if not self.is_trained: return None
        self.model.eval()
        features_np = np.array(features).reshape(1, -1)
        with torch.no_grad():
            features_scaled = self.scaler.transform(features_np)
            prob = self.model(torch.FloatTensor(features_scaled)).item()
        pca_coords = self.pca.transform(features_scaled)[0].tolist()
        return {"prediction": self.target_names[1] if prob >= 0.5 else self.target_names[0], "probability": prob, "pca": pca_coords}

    def _fig_to_b64(self, fig):
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches='tight', facecolor='white')
        plt.close(fig)
        return base64.b64encode(buf.getvalue()).decode('utf-8')

    def get_pca_plot(self):
        fig, ax = plt.subplots(figsize=(8,6))
        ax.scatter(self.X_train_pca[:,0], self.X_train_pca[:,1], c=self.y_train, cmap='coolwarm', alpha=0.6, label='Train')
        ax.scatter(self.X_test_pca[:,0], self.X_test_pca[:,1], c=self.y_test, cmap='coolwarm', marker='^', edgecolors='black', label='Test')
        ax.set_title(f"PCA: {self.dataset_name}"); ax.legend()
        return self._fig_to_b64(fig)

    def get_training_plot(self):
        if not self.history["losses"]: return None
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ax1.plot(self.history["losses"], color='red'); ax1.set_title("Loss"); ax1.set_xlabel("Epoch")
        ax2.plot(self.history["accuracies"], color='green'); ax2.set_title("Accuracy"); ax2.set_xlabel("Epoch")
        return self._fig_to_b64(fig)

    def get_confusion_plot(self, cm):
        cm = np.array(cm)
        fig, ax = plt.subplots(figsize=(6,5))
        im = ax.imshow(cm, cmap='Blues')
        ax.set_xticks([0,1]); ax.set_yticks([0,1])
        ax.set_xticklabels(self.target_names); ax.set_yticklabels(self.target_names) # DYNAMIC LABELS
        ax.set_ylabel('True'); ax.set_xlabel('Predicted')
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i][j]), ha="center", va="center", color="white" if cm[i][j] > cm.max()/2 else "black", fontsize=16)
        return self._fig_to_b64(fig)

    def get_decision_boundary_plot(self):
        if not self.is_trained or self.pca_svc is None: return None
        fig, ax = plt.subplots(figsize=(8,6))
        x_min, x_max = self.X_train_pca[:, 0].min() - 1, self.X_train_pca[:, 0].max() + 1
        y_min, y_max = self.X_train_pca[:, 1].min() - 1, self.X_train_pca[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
        Z = self.pca_svc.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        ax.contourf(xx, yy, Z, alpha=0.2, cmap='coolwarm')
        ax.contour(xx, yy, Z, colors='black', linewidths=1.5, linestyles='--')
        ax.scatter(self.X_train_pca[:,0], self.X_train_pca[:,1], c=self.y_train, cmap='coolwarm', edgecolors='black', s=40, label='Train Data')
        ax.scatter(self.X_test_pca[:,0], self.X_test_pca[:,1], c=self.y_test, cmap='coolwarm', marker='^', edgecolors='black', s=60, label='Test Data')
        ax.set_title(f"Decision Boundary: {self.dataset_name}"); ax.legend()
        return self._fig_to_b64(fig)

manager = ModelManager()
manager.load_data("breast_cancer") # Default startup dataset