import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import joblib
import os

from models.base_model import BaseModel

class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x)

class MLPModel(BaseModel):
    def __init__(self, input_dim, lr=0.001, epochs=20, batch_size=32):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = MLP(input_dim).to(self.device)
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.history = {"train": [], "val": []}

    def fit(self, X_train, y_train, X_val, y_val):
        print(self)
        X_train = torch.tensor(X_train.values, dtype=torch.float32)
        y_train = torch.tensor(y_train.values.reshape(-1, 1), dtype=torch.float32)

        X_val = torch.tensor(X_val.values, dtype=torch.float32)
        y_val = torch.tensor(y_val.values.reshape(-1, 1), dtype=torch.float32)

        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=self.batch_size)

        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        for epoch in range(self.epochs):
            self.model.train()
            train_loss = 0
            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                preds = self.model(xb)
                loss = criterion(preds, yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    preds = self.model(xb)
                    loss = criterion(preds, yb)
                    val_loss += loss.item()

            self.history["train"].append(train_loss / len(train_loader))
            self.history["val"].append(val_loss / len(val_loader))

            print(f"Epoch {epoch+1}/{self.epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")

        plt.figure()
        plt.plot(self.history["train"], label="Train Loss")
        plt.plot(self.history["val"], label="Validation Loss")
        plt.legend()
        plt.title("MLP Loss Curve")
        os.makedirs("models/mlp", exist_ok=True)
        plt.savefig("models/mlp/loss_curve.png")

    def predict(self, X):
        X = torch.tensor(X.values, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            preds = self.model(X)
        return (preds.cpu().numpy() > 0.5).astype(int)

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def __str__(self):
        return "MLPModel"
