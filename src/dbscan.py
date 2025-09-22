
import pandas as pd
from sklearn.cluster import DBSCAN
import joblib
import os

class DBSCANClusterer:
    def __init__(self, eps=1.25, min_samples=5, model_path="models/dbscan_model.pkl"):
        
        self.eps = eps
        self.min_samples = min_samples
        self.model_path = model_path
        self.model = None
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)

    def train(self, X: pd.DataFrame):
        
        self.model = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        labels = self.model.fit_predict(X)
        return labels

    def save_model(self):
        
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call train() first.")
        joblib.dump(self.model, self.model_path)
        print(f"DBSCAN model saved to {self.model_path}")

    def load_model(self):
        """
        Loads a pre-trained DBSCAN model from the specified path.
        """
        self.model = joblib.load(self.model_path)
        print(f"DBSCAN model loaded from {self.model_path}")
        return self.model