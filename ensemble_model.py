# ensemble_model.py
import numpy as np

class EnsembleModel:
    def __init__(self, models):
        self.models = models

    def predict(self, X):
        predictions = np.column_stack([model.predict(X) for model in self.models.values()])
        return np.mean(predictions, axis=1)