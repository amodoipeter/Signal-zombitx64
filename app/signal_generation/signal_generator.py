"""
Signal Generation Engine for the AI Signal Provider.
This module handles the AI-based generation of trading signals.
"""

import os
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from dotenv import load_dotenv

# Configure logging
logger = logging.getLogger(__name__)

class SignalGenerator:
    """
    A class to generate trading signals using machine learning models.
    Supports both LSTM and other time-series analysis techniques.
    """
    
    def __init__(self, model_path=None):
        """
        Initialize the SignalGenerator.
        
        Args:
            model_path (str, optional): Path to a pre-trained model file
        """
        load_dotenv()
        
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.threshold = float(os.getenv("SIGNAL_CONFIDENCE_THRESHOLD", 0.75))
        
        if model_path and os.path.exists(model_path):
            self._load_model(model_path)
    
    def _load_model(self, model_path):
        """
        Load a pre-trained model from disk.
        
        Args:
            model_path (str): Path to the model file
        """
        try:
            self.model = load_model(model_path)
            logger.info(f"Successfully loaded model from {model_path}")
        except Exception as e:
            logger.error(f"Error loading model from {model_path}: {str(e)}")
            raise
    
    def _prepare_data(self, df, sequence_length=60):
        """
        Prepare data for the LSTM model by creating sequences.
        
        Args:
            df (pd.DataFrame): DataFrame containing market data
            sequence_length (int): Number of time steps to look back
            
        Returns:
            tuple: (X, y) where X is the input sequences and y is the target values
        """
        # Get the features we care about
        data = df[['open', 'high', 'low', 'close', 'volume']].values
        
        # Scale the data
        scaled_data = self.scaler.fit_transform(data)
        
        X, y = [], []
        
        # Create sequences
        for i in range(sequence_length, len(scaled_data)):
            X.append(scaled_data[i-sequence_length:i])
            # Predict the direction of the close price (up=1, down=0)
            y.append(1 if scaled_data[i, 3] > scaled_data[i-1, 3] else 0)
        
        return np.array(X), np.array(y)
    
    def train_model(self, df, epochs=50, batch_size=32, sequence_length=60):
        """
        Train a new LSTM model on the provided market data.
        
        Args:
            df (pd.DataFrame): DataFrame containing market data
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            sequence_length (int): Number of time steps to look back
            
        Returns:
            tuple: (history, model) - Training history and the trained model
        """
        logger.info("Training new LSTM model...")
        
        X, y = self._prepare_data(df, sequence_length)
        
        # Split data for training and validation
        split = int(0.8 * len(X))
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]
        
        # Build LSTM model
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(units=1, activation='sigmoid'))
        
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        # Train the model
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            verbose=1
        )
        
        self.model = model
        logger.info("Model training completed")
        
        return history, model
    
    def save_model(self, file_path):
        """
        Save the trained model to disk.
        
        Args:
            file_path (str): Path to save the model to
        """
        if self.model is None:
            raise ValueError("No model to save. Train or load a model first.")
        
        try:
            self.model.save(file_path)
            logger.info(f"Model saved to {file_path}")
        except Exception as e:
            logger.error(f"Error saving model to {file_path}: {str(e)}")
            raise
    
    def generate_signal(self, df, symbol):
        """
        Generate a trading signal for the given market data.
        
        Args:
            df (pd.DataFrame): DataFrame containing recent market data
            symbol (str): Trading symbol (e.g., 'BTC/USDT')
            
        Returns:
            dict: A dictionary containing the signal information
        """
        if self.model is None:
            raise ValueError("No model loaded. Train or load a model first.")
        
        logger.info(f"Generating signal for {symbol}...")
        
        X, _ = self._prepare_data(df)
        
        # Use the last sequence for prediction
        if len(X) == 0:
            logger.warning(f"Not enough data to generate signal for {symbol}")
            return None
            
        X_last = X[-1:]
        
        # Predict
        prediction = self.model.predict(X_last)[0][0]
        
        # Determine signal type based on prediction
        signal_type = "NEUTRAL"
        confidence = 0.5
        
        if prediction > self.threshold:
            signal_type = "BUY"
            confidence = float(prediction)
        elif prediction < (1 - self.threshold):
            signal_type = "SELL"
            confidence = float(1 - prediction)
        
        # Create signal
        signal = {
            "symbol": symbol,
            "signal": signal_type,
            "confidence": confidence,
            "timestamp": datetime.now().isoformat(),
            "price": df['close'].iloc[-1],
            "volume": df['volume'].iloc[-1]
        }
        
        logger.info(f"Generated {signal_type} signal for {symbol} with confidence {confidence:.2f}")
        return signal
    
    def evaluate_model(self, df, sequence_length=60):
        """
        Evaluate the model performance on historical data.
        
        Args:
            df (pd.DataFrame): DataFrame containing market data
            sequence_length (int): Number of time steps to look back
            
        Returns:
            dict: Evaluation metrics
        """
        if self.model is None:
            raise ValueError("No model loaded. Train or load a model first.")
        
        X, y = self._prepare_data(df, sequence_length)
        
        # Evaluate
        loss, accuracy = self.model.evaluate(X, y, verbose=0)
        
        # Make predictions
        predictions = self.model.predict(X)
        
        # Calculate additional metrics
        predicted_classes = (predictions > 0.5).astype(int).reshape(-1)
        
        # Calculate true positives, false positives, etc.
        tp = np.sum((predicted_classes == 1) & (y == 1))
        fp = np.sum((predicted_classes == 1) & (y == 0))
        tn = np.sum((predicted_classes == 0) & (y == 0))
        fn = np.sum((predicted_classes == 0) & (y == 1))
        
        # Calculate precision, recall, f1-score
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics = {
            "loss": float(loss),
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1)
        }
        
        logger.info(f"Model evaluation metrics: {metrics}")
        return metrics


# Example usage
if __name__ == "__main__":
    from app.data_collection.market_data import MarketDataCollector
    
    # Collect some data
    collector = MarketDataCollector()
    df = collector.fetch_historical_data("BTC/USDT", days_back=60)
    
    # Initialize signal generator
    signal_gen = SignalGenerator()
    
    # Train the model
    history, model = signal_gen.train_model(df)
    
    # Save the model
    signal_gen.save_model("models/btc_model.h5")
    
    # Generate a signal
    signal = signal_gen.generate_signal(df, "BTC/USDT")
    print(f"Signal: {signal}")
    
    # Evaluate the model
    metrics = signal_gen.evaluate_model(df)
    print(f"Metrics: {metrics}")
