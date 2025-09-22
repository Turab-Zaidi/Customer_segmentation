import pandas as pd
from sklearn.cluster import KMeans
import joblib

class KMeansClusterer:
    def __init__(self, n_clusters=3, random_state=42, model_path="models/kmeans_model.pkl"):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.model_path = model_path
        self.model = None

    def train(self, X: pd.DataFrame):

        self.model = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
        labels = self.model.fit_predict(X)
        return labels

    def save_model(self):

        if self.model is None:
            raise ValueError("Model not trained yet!")
        joblib.dump(self.model, self.model_path)
        print(f"KMeans model saved to {self.model_path}")

    def load_model(self):

        self.model = joblib.load(self.model_path)
        return self.model

    def assign_clusters(self, X: pd.DataFrame):

        if self.model is None:
            raise ValueError("Model not trained or loaded!")
        labels = self.model.predict(X)
        return labels