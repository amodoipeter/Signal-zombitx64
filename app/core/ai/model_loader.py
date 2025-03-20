import os
import logging
import pickle
from typing import Any, Optional
import numpy as np

logger = logging.getLogger(__name__)

# Directory where models are stored
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

# Placeholder for ML model in development environments
class DummyModel:
    """Dummy model for development/testing purposes."""
    def predict(self, X: np.ndarray):
        """Return random prediction for testing."""
        predictions = []
        for _ in range(len(X)):
            # Random prediction: 0 for buy, 1 for sell
            pred = np.random.choice([0, 1])
            # Random confidence between 60-100%
            confidence = np.random.uniform(60, 100)
            predictions.append((pred, confidence))
        return predictions
    
    def predict_proba(self, X: np.ndarray):
        """Return random probabilities for testing."""
        probas = []
        for _ in range(len(X)):
            # Random probabilities that sum to 1
            buy_prob = np.random.uniform(0, 1)
            sell_prob = 1 - buy_prob
            probas.append([buy_prob, sell_prob])
        return np.array(probas)

def load_model(model_name: str = "signal_model_v1.pkl") -> Any:
    """
    Load the ML model from disk.
    
    In production, this would load a trained model.
    For development, it returns a dummy model.
    """
    model_path = os.path.join(MODEL_DIR, model_name)
    
    # Check if we're in development mode or model doesn't exist
    if os.getenv("ENV", "development") == "development" or not os.path.exists(model_path):
        logger.warning("Using dummy model for development")
        return DummyModel()
    
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        logger.info(f"Model {model_name} loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Error loading model {model_name}: {str(e)}")
        logger.warning("Falling back to dummy model")
        return DummyModel()
