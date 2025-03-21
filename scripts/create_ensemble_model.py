#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to create and save an ensemble model combining LSTM, GRU, and traditional ML models.
"""

import os
import sys
import pickle
import logging
from typing import List, Dict, Any
import argparse

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.ai.model_loader import (
    LSTMModel, GRUModel, CNNLSTMModel, SklearnModel, 
    EnsembleModel, ModelType, load_model
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Models directory
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                         "app", "core", "ai", "models")

def create_ensemble():
    """Create and save an ensemble model from multiple individual models."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # Load individual models
    models = []
    weights = []
    
    # Try to load LSTM model
    try:
        logger.info("Loading LSTM model...")
        lstm_model = load_model(ModelType.LSTM)
        models.append(lstm_model)
        weights.append(1.2)  # Higher weight for LSTM as it's generally better for time series
        logger.info("LSTM model loaded successfully")
    except Exception as e:
        logger.warning(f"Could not load LSTM model: {str(e)}")
    
    # Try to load GRU model
    try:
        logger.info("Loading GRU model...")
        gru_model = load_model(ModelType.GRU)
        models.append(gru_model)
        weights.append(1.1)
        logger.info("GRU model loaded successfully")
    except Exception as e:
        logger.warning(f"Could not load GRU model: {str(e)}")
    
    # Try to load RandomForest model
    try:
        logger.info("Loading RandomForest model...")
        rf_model = load_model(ModelType.RANDOM_FOREST)
        models.append(rf_model)
        weights.append(0.9)
        logger.info("RandomForest model loaded successfully")
    except Exception as e:
        logger.warning(f"Could not load RandomForest model: {str(e)}")
    
    # Try to load GradientBoosting model
    try:
        logger.info("Loading GradientBoosting model...")
        gb_model = load_model(ModelType.GRADIENT_BOOST)
        models.append(gb_model)
        weights.append(1.0)
        logger.info("GradientBoosting model loaded successfully")
    except Exception as e:
        logger.warning(f"Could not load GradientBoosting model: {str(e)}")
    
    if not models:
        logger.error("No models could be loaded. Using dummy model.")
        dummy_model = load_model(ModelType.DUMMY)
        models = [dummy_model]
        weights = [1.0]
    
    # Create ensemble
    ensemble = EnsembleModel(models, weights)
    
    # Create configuration for saving
    ensemble_config = {
        "models": [
            {
                "type": model.__class__.__name__.replace("Model", "").lower(),
                "path": f"latest_{model.__class__.__name__.replace('Model', '').lower()}_model",
                "weight": weight
            }
            for model, weight in zip(models, weights)
        ]
    }
    
    # Save ensemble configuration
    ensemble_path = os.path.join(MODELS_DIR, "latest_ensemble_model.pkl")
    with open(ensemble_path, 'wb') as f:
        pickle.dump(ensemble_config, f)
    
    logger.info(f"Ensemble model saved to {ensemble_path}")
    logger.info(f"Ensemble configuration: {len(models)} models with weights {weights}")
    
    return ensemble

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create ensemble model from individual models")
    parser.add_argument("--test", action="store_true", help="Test the created ensemble model")
    args = parser.parse_args()
    
    ensemble = create_ensemble()
    
    if args.test:
        import numpy as np
        # Create sample data
        X = np.random.random((5, 60, 20))  # 5 samples, 60 timesteps, 20 features
        
        # Test ensemble
        predictions = ensemble.predict(X)
        print("\nSample predictions:")
        for i, (cls, conf) in enumerate(predictions):
            print(f"Sample {i}: Class {cls} (Buy: 0, Sell: 1, Hold: 2) with confidence {conf:.2f}%")
