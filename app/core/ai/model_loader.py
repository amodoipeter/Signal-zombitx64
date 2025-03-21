import os
import logging
import pickle
import numpy as np
import pandas as pd
from typing import Any, Optional, Dict, List, Tuple, Union
import joblib
from datetime import datetime
from enum import Enum

# Machine Learning imports
try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, Dense, Dropout, GRU, Bidirectional, Conv1D, MaxPooling1D, TimeDistributed
    ML_LIBRARIES_AVAILABLE = True
except ImportError:
    ML_LIBRARIES_AVAILABLE = False

# RL imports
try:
    import gym
    from stable_baselines3 import A2C, PPO, DQN
    from stable_baselines3.common.vec_env import DummyVecEnv
    RL_LIBRARIES_AVAILABLE = True
except ImportError:
    RL_LIBRARIES_AVAILABLE = False

logger = logging.getLogger(__name__)

# Directory where models are stored
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODEL_DIR, exist_ok=True)

class ModelType(str, Enum):
    LSTM = "lstm"
    GRU = "gru"
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOST = "gradient_boost"
    TRANSFORMER = "transformer"
    ENSEMBLE = "ensemble"
    RL = "reinforcement_learning" 
    CNN_LSTM = "cnn_lstm"
    DUMMY = "dummy"

class DummyModel:
    """Dummy model for development/testing purposes."""
    def predict(self, X: np.ndarray):
        """Return random prediction for testing."""
        predictions = []
        for _ in range(len(X)):
            # Random prediction: 0 for buy, 1 for sell, 2 for hold
            pred = np.random.choice([0, 1, 2])
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
            sell_prob = np.random.uniform(0, 1-buy_prob)
            hold_prob = 1 - buy_prob - sell_prob
            probas.append([buy_prob, sell_prob, hold_prob])
        return np.array(probas)

class LSTMModel:
    """LSTM model for time series forecasting."""
    def __init__(self, model_path: str = None):
        if not ML_LIBRARIES_AVAILABLE:
            raise ImportError("TensorFlow/Keras is required for LSTM model")
        
        if model_path and os.path.exists(model_path):
            self.model = load_model(model_path)
            # Load scaler if available
            scaler_path = model_path.replace('.h5', '_scaler.pkl')
            if os.path.exists(scaler_path):
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
            else:
                self.scaler = StandardScaler()
        else:
            logger.warning(f"LSTM model not found at {model_path}, initializing new model")
            self._initialize_model()
            self.scaler = StandardScaler()
    
    def _initialize_model(self, input_shape: Tuple = (60, 50)):
        """Initialize a new LSTM model."""
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(units=25))
        model.add(Dense(units=3, activation='softmax'))  # 3 outputs: buy, sell, hold
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        self.model = model
    
    def predict(self, X: np.ndarray):
        """Predict using the LSTM model."""
        # Ensure X is properly shaped for LSTM (samples, time steps, features)
        if len(X.shape) == 2:
            X = X.reshape((X.shape[0], 1, X.shape[1]))
        
        # Scale features
        X_scaled = self.scaler.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
        
        # Get predictions
        probs = self.model.predict(X_scaled)
        
        # Get classes and confidence
        predictions = []
        for prob in probs:
            cls = np.argmax(prob)
            confidence = prob[cls] * 100
            predictions.append((cls, confidence))
        
        return predictions
    
    def predict_proba(self, X: np.ndarray):
        """Return class probabilities."""
        # Ensure X is properly shaped for LSTM (samples, time steps, features)
        if len(X.shape) == 2:
            X = X.reshape((X.shape[0], 1, X.shape[1]))
        
        # Scale features
        X_scaled = self.scaler.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
        
        return self.model.predict(X_scaled)

class GRUModel:
    """GRU model for time series forecasting."""
    def __init__(self, model_path: str = None):
        if not ML_LIBRARIES_AVAILABLE:
            raise ImportError("TensorFlow/Keras is required for GRU model")
        
        if model_path and os.path.exists(model_path):
            self.model = load_model(model_path)
            # Load scaler if available
            scaler_path = model_path.replace('.h5', '_scaler.pkl')
            if os.path.exists(scaler_path):
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
            else:
                self.scaler = StandardScaler()
        else:
            logger.warning(f"GRU model not found at {model_path}, initializing new model")
            self._initialize_model()
            self.scaler = StandardScaler()
    
    def _initialize_model(self, input_shape: Tuple = (60, 50)):
        """Initialize a new GRU model."""
        model = Sequential()
        model.add(GRU(units=50, return_sequences=True, input_shape=input_shape))
        model.add(Dropout(0.2))
        model.add(GRU(units=50, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(units=25))
        model.add(Dense(units=3, activation='softmax'))  # 3 outputs: buy, sell, hold
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        self.model = model
    
    def predict(self, X: np.ndarray):
        """Predict using the GRU model."""
        # Ensure X is properly shaped for GRU (samples, time steps, features)
        if len(X.shape) == 2:
            X = X.reshape((X.shape[0], 1, X.shape[1]))
        
        # Scale features
        X_scaled = self.scaler.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
        
        # Get predictions
        probs = self.model.predict(X_scaled)
        
        # Get classes and confidence
        predictions = []
        for prob in probs:
            cls = np.argmax(prob)
            confidence = prob[cls] * 100
            predictions.append((cls, confidence))
        
        return predictions
    
    def predict_proba(self, X: np.ndarray):
        """Return class probabilities."""
        # Ensure X is properly shaped for GRU (samples, time steps, features)
        if len(X.shape) == 2:
            X = X.reshape((X.shape[0], 1, X.shape[1]))
        
        # Scale features
        X_scaled = self.scaler.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
        
        return self.model.predict(X_scaled)

class CNNLSTMModel:
    """CNN-LSTM hybrid model for time series forecasting."""
    def __init__(self, model_path: str = None):
        if not ML_LIBRARIES_AVAILABLE:
            raise ImportError("TensorFlow/Keras is required for CNN-LSTM model")
        
        if model_path and os.path.exists(model_path):
            self.model = load_model(model_path)
            # Load scaler if available
            scaler_path = model_path.replace('.h5', '_scaler.pkl')
            if os.path.exists(scaler_path):
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
            else:
                self.scaler = StandardScaler()
        else:
            logger.warning(f"CNN-LSTM model not found at {model_path}, initializing new model")
            self._initialize_model()
            self.scaler = StandardScaler()
    
    def _initialize_model(self, input_shape: Tuple = (60, 50)):
        """Initialize a new CNN-LSTM hybrid model."""
        model = Sequential()
        
        # CNN layers for feature extraction
        model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape))
        model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Dropout(0.2))
        
        # LSTM layers for sequence learning
        model.add(LSTM(units=100, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50, return_sequences=False))
        model.add(Dropout(0.2))
        
        # Output layers
        model.add(Dense(units=25, activation='relu'))
        model.add(Dense(units=3, activation='softmax'))  # 3 outputs: buy, sell, hold
        
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        self.model = model
    
    def predict(self, X: np.ndarray):
        """Predict using the CNN-LSTM model."""
        # Ensure X is properly shaped for CNN-LSTM (samples, time steps, features)
        if len(X.shape) == 2:
            X = X.reshape((X.shape[0], 1, X.shape[1]))
        
        # Scale features
        X_scaled = self.scaler.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
        
        # Get predictions
        probs = self.model.predict(X_scaled)
        
        # Get classes and confidence
        predictions = []
        for prob in probs:
            cls = np.argmax(prob)
            confidence = prob[cls] * 100
            predictions.append((cls, confidence))
        
        return predictions
    
    def predict_proba(self, X: np.ndarray):
        """Return class probabilities."""
        # Ensure X is properly shaped for CNN-LSTM (samples, time steps, features)
        if len(X.shape) == 2:
            X = X.reshape((X.shape[0], 1, X.shape[1]))
        
        # Scale features
        X_scaled = self.scaler.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
        
        return self.model.predict(X_scaled)

class RLModel:
    """Reinforcement Learning model for trading."""
    def __init__(self, model_path: str = None):
        if not RL_LIBRARIES_AVAILABLE:
            raise ImportError("stable-baselines3 is required for RL model")
        
        if model_path and os.path.exists(model_path):
            try:
                self.model = PPO.load(model_path)
                logger.info(f"Loaded PPO model from {model_path}")
            except Exception:
                try:
                    self.model = A2C.load(model_path)
                    logger.info(f"Loaded A2C model from {model_path}")
                except Exception:
                    try:
                        self.model = DQN.load(model_path)
                        logger.info(f"Loaded DQN model from {model_path}")
                    except Exception as e:
                        logger.error(f"Could not load RL model: {str(e)}")
                        self.model = None
        else:
            logger.warning(f"RL model not found at {model_path}, using dummy predictions")
            self.model = None
    
    def predict(self, X: np.ndarray):
        """Predict using the RL model."""
        if self.model is None:
            # Return dummy predictions
            dummy = DummyModel()
            return dummy.predict(X)
        
        # Convert features to the format expected by RL model
        observations = self._preprocess_features(X)
        
        # Get actions and values from model
        actions, values = [], []
        for obs in observations:
            action, _ = self.model.predict(obs, deterministic=True)
            value = float(self.model.policy.value_net(obs).cpu().data.numpy()[0])
            actions.append(action)
            values.append(value)
        
        # Map actions to buy/sell/hold predictions
        predictions = []
        for action, value in zip(actions, values):
            # Map action to class (0-buy, 1-sell, 2-hold)
            if action == 0:
                cls = 0  # buy
            elif action == 1:
                cls = 1  # sell
            else:
                cls = 2  # hold
            
            # Calculate confidence (normalize value to 0-100%)
            confidence = min(max((value + 2) * 20, 60), 100)  # Map value to 60-100% range
            predictions.append((cls, confidence))
        
        return predictions
    
    def _preprocess_features(self, X: np.ndarray):
        """Preprocess features for RL model input."""
        # Normalize if needed
        X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
        return X

class SklearnModel:
    """Wrapper for sklearn models like RandomForest and GradientBoosting."""
    def __init__(self, model_path: str = None):
        if model_path and os.path.exists(model_path):
            self.model = joblib.load(model_path)
            # Load scaler if available
            scaler_path = model_path.replace('.joblib', '_scaler.joblib')
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
            else:
                self.scaler = StandardScaler()
        else:
            logger.warning(f"Sklearn model not found at {model_path}, using dummy model")
            self.model = None
            self.scaler = StandardScaler()
    
    def predict(self, X: np.ndarray):
        """Predict using the sklearn model."""
        if self.model is None:
            dummy = DummyModel()
            return dummy.predict(X)
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Get predictions
        if hasattr(self.model, 'predict_proba'):
            probs = self.model.predict_proba(X_scaled)
            
            # Get classes and confidence
            predictions = []
            for prob in probs:
                cls = np.argmax(prob)
                confidence = prob[cls] * 100
                predictions.append((cls, confidence))
            return predictions
        else:
            # If model doesn't support predict_proba, return class with 80% confidence
            classes = self.model.predict(X_scaled)
            return [(cls, 80.0) for cls in classes]
    
    def predict_proba(self, X: np.ndarray):
        """Return class probabilities."""
        if self.model is None or not hasattr(self.model, 'predict_proba'):
            dummy = DummyModel()
            return dummy.predict_proba(X)
            
        # Scale features
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)

class EnsembleModel:
    """Ensemble model combining multiple models for better predictions."""
    def __init__(self, models: List[Any] = None, weights: List[float] = None):
        self.models = models or []
        self.weights = weights or [1.0] * len(self.models)
        
        # Normalize weights
        total = sum(self.weights)
        self.weights = [w / total for w in self.weights]
    
    def add_model(self, model: Any, weight: float = 1.0):
        """Add a model to the ensemble."""
        self.models.append(model)
        self.weights.append(weight)
        
        # Renormalize weights
        total = sum(self.weights)
        self.weights = [w / total for w in self.weights]
    
    def predict(self, X: np.ndarray):
        """Predict using the ensemble of models."""
        if not self.models:
            dummy = DummyModel()
            return dummy.predict(X)
        
        all_predictions = []
        
        # Get predictions from all models
        for model in self.models:
            try:
                preds = model.predict(X)
                all_predictions.append(preds)
            except Exception as e:
                logger.error(f"Error getting predictions from model {model.__class__.__name__}: {str(e)}")
        
        # Combine predictions using weights
        final_predictions = []
        for i in range(len(X)):
            # Count votes for each class
            class_votes = {0: 0.0, 1: 0.0, 2: 0.0}  # buy, sell, hold
            confidence_sum = {0: 0.0, 1: 0.0, 2: 0.0}  # sum of confidence scores
            
            # Collect votes from all models
            for m_idx, model_preds in enumerate(all_predictions):
                if i < len(model_preds):
                    cls, conf = model_preds[i]
                    class_votes[cls] += self.weights[m_idx]
                    confidence_sum[cls] += conf * self.weights[m_idx]
            
            # Find class with highest weighted votes
            max_votes = max(class_votes.values())
            if max_votes > 0:
                # There might be a tie, choose class with highest confidence
                max_classes = [c for c, v in class_votes.items() if v == max_votes]
                if len(max_classes) == 1:
                    best_class = max_classes[0]
                else:
                    # Tiebreaker: choose class with highest confidence
                    best_class = max(max_classes, key=lambda c: confidence_sum.get(c, 0))
                
                # Calculate weighted average confidence for the chosen class
                if class_votes[best_class] > 0:
                    confidence = confidence_sum[best_class] / class_votes[best_class]
                else:
                    confidence = 70  # Default confidence
            else:
                best_class = 2  # Hold by default
                confidence = 70  # Default confidence
            
            final_predictions.append((best_class, confidence))
        
        return final_predictions
    
    def predict_proba(self, X: np.ndarray):
        """Predict class probabilities using the ensemble."""
        if not self.models:
            dummy = DummyModel()
            return dummy.predict_proba(X)
        
        all_probas = []
        
        # Get probabilities from all models
        for model in self.models:
            try:
                if hasattr(model, 'predict_proba'):
                    probas = model.predict_proba(X)
                    all_probas.append(probas)
            except Exception as e:
                logger.error(f"Error getting probabilities from model {model.__class__.__name__}: {str(e)}")
        
        if not all_probas:
            dummy = DummyModel()
            return dummy.predict_proba(X)
            
        # Combine probabilities using weights
        weighted_sum_probas = np.zeros((len(X), 3))  # 3 classes: buy, sell, hold
        
        for m_idx, model_probas in enumerate(all_probas):
            weighted_sum_probas += model_probas * self.weights[m_idx]
        
        # Normalize
        row_sums = weighted_sum_probas.sum(axis=1)
        normalized_probas = weighted_sum_probas / row_sums[:, np.newaxis]
        
        return normalized_probas

def load_model_by_type(model_type: ModelType, model_name: str = None) -> Any:
    """
    Load a model of the specified type.
    
    Args:
        model_type: Type of model to load
        model_name: Specific model name/version
        
    Returns:
        Loaded model instance
    """
    if model_name:
        model_path = os.path.join(MODEL_DIR, model_name)
    else:
        # Use default name based on type
        model_filename = f"latest_{model_type.value}_model"
        if model_type in [ModelType.LSTM, ModelType.GRU, ModelType.CNN_LSTM]:
            model_filename += ".h5"
        elif model_type in [ModelType.RANDOM_FOREST, ModelType.GRADIENT_BOOST]:
            model_filename += ".joblib"
        elif model_type == ModelType.RL:
            model_filename += ".zip"
        elif model_type == ModelType.ENSEMBLE:
            model_filename += ".pkl"
        
        model_path = os.path.join(MODEL_DIR, model_filename)
    
    try:
        if model_type == ModelType.LSTM:
            return LSTMModel(model_path)
            
        elif model_type == ModelType.GRU:
            return GRUModel(model_path)
            
        elif model_type == ModelType.CNN_LSTM:
            return CNNLSTMModel(model_path)
        
        elif model_type == ModelType.RL:
            return RLModel(model_path)
        
        elif model_type == ModelType.RANDOM_FOREST or model_type == ModelType.GRADIENT_BOOST:
            return SklearnModel(model_path)
        
        elif model_type == ModelType.ENSEMBLE:
            # Load ensemble configuration
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    ensemble_config = pickle.load(f)
                    
                ensemble = EnsembleModel()
                
                # Load each model in the ensemble
                for model_info in ensemble_config.get('models', []):
                    model_type_str = model_info.get('type')
                    model_path = model_info.get('path')
                    weight = model_info.get('weight', 1.0)
                    
                    if model_type_str and model_path:
                        try:
                            model = load_model_by_type(ModelType(model_type_str), model_path)
                            ensemble.add_model(model, weight)
                        except Exception as e:
                            logger.error(f"Error loading ensemble model component: {str(e)}")
                
                return ensemble
            else:
                logger.warning(f"Ensemble model not found at {model_path}, creating empty ensemble")
                return EnsembleModel()
        
        else:
            logger.warning(f"Unknown model type: {model_type}, using dummy model")
            return DummyModel()
            
    except Exception as e:
        logger.error(f"Error loading model {model_type} from {model_path}: {str(e)}")
        logger.warning(f"Falling back to dummy model")
        return DummyModel()

def load_model(model_type: ModelType = ModelType.ENSEMBLE, model_name: str = None) -> Any:
    """
    Main function to load an AI model for trading signal prediction.
    
    Args:
        model_type: Type of model to load (defaults to ensemble)
        model_name: Specific model name or path
        
    Returns:
        Model instance with predict() and predict_proba() methods
    """
    try:
        logger.info(f"Loading {model_type.value} model")
        return load_model_by_type(model_type, model_name)
    except Exception as e:
        logger.error(f"Failed to load {model_type.value} model: {str(e)}")
        logger.warning("Using dummy model instead")
        return DummyModel()
