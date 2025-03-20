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
        
        # Calculate technical indicators
        df_with_indicators = self.calculate_technical_indicators(df)
        
        # Use both AI model prediction and technical indicators for signal generation
        X, _ = self._prepare_data(df)
        
        # Use the last sequence for prediction
        if len(X) == 0:
            logger.warning(f"Not enough data to generate signal for {symbol}")
            return None
            
        X_last = X[-1:]
        
        # Get AI model prediction
        prediction = self.model.predict(X_last)[0][0]
        
        # Get technical indicator signals
        rsi_signal = self._get_rsi_signal(df_with_indicators)
        bollinger_signal = self._get_bollinger_signal(df_with_indicators)
        macd_signal = self._get_macd_signal(df_with_indicators)
        ichimoku_signal = self._get_ichimoku_signal(df_with_indicators)
        
        # Combine signals (simple majority voting)
        tech_signals = [rsi_signal, bollinger_signal, macd_signal, ichimoku_signal]
        buy_count = tech_signals.count("BUY")
        sell_count = tech_signals.count("SELL")
        neutral_count = tech_signals.count("NEUTRAL")
        
        # Final decision logic
        signal_type = "NEUTRAL"
        confidence = 0.5
        
        # AI model has high confidence
        if prediction > self.threshold:
            signal_type = "BUY"
            confidence = float(prediction)
            # Boost confidence if technical indicators agree
            if buy_count >= 2:
                confidence = min(confidence + 0.1, 0.99)
            # Reduce confidence if technical indicators disagree strongly
            elif sell_count >= 3:
                confidence = max(confidence - 0.2, 0.5)
                signal_type = "NEUTRAL"  # Override to neutral due to conflict
        elif prediction < (1 - self.threshold):
            signal_type = "SELL"
            confidence = float(1 - prediction)
            # Boost confidence if technical indicators agree
            if sell_count >= 2:
                confidence = min(confidence + 0.1, 0.99)
            # Reduce confidence if technical indicators disagree strongly
            elif buy_count >= 3:
                confidence = max(confidence - 0.2, 0.5)
                signal_type = "NEUTRAL"  # Override to neutral due to conflict
        else:
            # If AI model isn't confident, rely more on technical indicators
            if buy_count >= 3:
                signal_type = "BUY"
                confidence = 0.6 + (buy_count * 0.05)
            elif sell_count >= 3:
                signal_type = "SELL"
                confidence = 0.6 + (sell_count * 0.05)
        
        # Include potential take profit and stop loss levels
        tp_level, sl_level = self._calculate_tp_sl_levels(df_with_indicators, signal_type)
        
        # Create signal
        signal = {
            "symbol": symbol,
            "signal": signal_type,
            "confidence": confidence,
            "timestamp": datetime.now().isoformat(),
            "price": df['close'].iloc[-1],
            "volume": df['volume'].iloc[-1],
            "take_profit": tp_level,
            "stop_loss": sl_level,
            "technical_indicators": {
                "rsi": df_with_indicators['rsi'].iloc[-1],
                "macd": df_with_indicators['macd'].iloc[-1],
                "macd_signal": df_with_indicators['macd_signal'].iloc[-1],
                "bollinger_upper": df_with_indicators['bollinger_upper'].iloc[-1],
                "bollinger_lower": df_with_indicators['bollinger_lower'].iloc[-1],
                "ichimoku": {
                    "tenkan_sen": df_with_indicators['tenkan_sen'].iloc[-1],
                    "kijun_sen": df_with_indicators['kijun_sen'].iloc[-1]
                }
            }
        }
        
        logger.info(f"Generated {signal_type} signal for {symbol} with confidence {confidence:.2f}")
        return signal
    
    def _get_rsi_signal(self, df):
        """Get trading signal based on RSI"""
        rsi = df['rsi'].iloc[-1]
        if rsi < 30:
            return "BUY"
        elif rsi > 70:
            return "SELL"
        return "NEUTRAL"
    
    def _get_bollinger_signal(self, df):
        """Get trading signal based on Bollinger Bands"""
        current_price = df['close'].iloc[-1]
        upper_band = df['bollinger_upper'].iloc[-1]
        lower_band = df['bollinger_lower'].iloc[-1]
        
        if current_price > upper_band:
            return "SELL"
        elif current_price < lower_band:
            return "BUY"
        return "NEUTRAL"
    
    def _get_macd_signal(self, df):
        """Get trading signal based on MACD"""
        macd = df['macd'].iloc[-1]
        macd_signal = df['macd_signal'].iloc[-1]
        
        # Check for crossover
        previous_macd = df['macd'].iloc[-2] if len(df) > 1 else 0
        previous_signal = df['macd_signal'].iloc[-2] if len(df) > 1 else 0
        
        if macd > macd_signal and previous_macd <= previous_signal:
            return "BUY"
        elif macd < macd_signal and previous_macd >= previous_signal:
            return "SELL"
        return "NEUTRAL"
    
    def _get_ichimoku_signal(self, df):
        """Get trading signal based on Ichimoku Cloud"""
        current_price = df['close'].iloc[-1]
        tenkan_sen = df['tenkan_sen'].iloc[-1]
        kijun_sen = df['kijun_sen'].iloc[-1]
        senkou_a = df['senkou_span_a'].iloc[-1]
        senkou_b = df['senkou_span_b'].iloc[-1]
        
        # Price above the cloud
        if current_price > max(senkou_a, senkou_b) and tenkan_sen > kijun_sen:
            return "BUY"
        # Price below the cloud
        elif current_price < min(senkou_a, senkou_b) and tenkan_sen < kijun_sen:
            return "SELL"
        return "NEUTRAL"
    
    def _calculate_tp_sl_levels(self, df, signal_type):
        """Calculate take profit and stop loss levels based on signal type and volatility"""
        current_price = df['close'].iloc[-1]
        
        # Calculate Average True Range (ATR) for dynamic TP/SL setting
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=14).mean().iloc[-1]
        
        # Set TP/SL based on signal type and ATR
        if signal_type == "BUY":
            take_profit = current_price + (atr * 3)  # 3x ATR for TP
            stop_loss = current_price - (atr * 1.5)  # 1.5x ATR for SL
        elif signal_type == "SELL":
            take_profit = current_price - (atr * 3)  # 3x ATR for TP
            stop_loss = current_price + (atr * 1.5)  # 1.5x ATR for SL
        else:
            take_profit = None
            stop_loss = None
        
        return take_profit, stop_loss
    
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
    
    def _calculate_rsi(self, prices, period=14):
        """
        Calculate the Relative Strength Index (RSI).
        
        Args:
            prices (pd.Series): Series of prices
            period (int): RSI period, default 14
            
        Returns:
            pd.Series: RSI values
        """
        delta = prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _calculate_ichimoku(self, df, tenkan_period=9, kijun_period=26, senkou_b_period=52):
        """
        Calculate Ichimoku Cloud indicator.
        
        Args:
            df (pd.DataFrame): DataFrame with OHLC data
            tenkan_period (int): Tenkan-sen period
            kijun_period (int): Kijun-sen period
            senkou_b_period (int): Senkou Span B period
            
        Returns:
            pd.DataFrame: DataFrame with Ichimoku values
        """
        # Calculate Tenkan-sen (Conversion Line)
        tenkan_high = df['high'].rolling(window=tenkan_period).max()
        tenkan_low = df['low'].rolling(window=tenkan_period).min()
        df['tenkan_sen'] = (tenkan_high + tenkan_low) / 2
        
        # Calculate Kijun-sen (Base Line)
        kijun_high = df['high'].rolling(window=kijun_period).max()
        kijun_low = df['low'].rolling(window=kijun_period).min()
        df['kijun_sen'] = (kijun_high + kijun_low) / 2
        
        # Calculate Senkou Span A (Leading Span A)
        df['senkou_span_a'] = ((df['tenkan_sen'] + df['kijun_sen']) / 2).shift(kijun_period)
        
        # Calculate Senkou Span B (Leading Span B)
        senkou_high = df['high'].rolling(window=senkou_b_period).max()
        senkou_low = df['low'].rolling(window=senkou_b_period).min()
        df['senkou_span_b'] = ((senkou_high + senkou_low) / 2).shift(kijun_period)
        
        # Calculate Chikou Span (Lagging Span)
        df['chikou_span'] = df['close'].shift(-kijun_period)
        
        return df
    
    def _calculate_bollinger_bands(self, prices, window=20, num_std=2):
        """
        Calculate Bollinger Bands.
        
        Args:
            prices (pd.Series): Series of prices
            window (int): Moving average window, default 20
            num_std (int): Number of standard deviations, default 2
            
        Returns:
            tuple: (middle_band, upper_band, lower_band)
        """
        middle_band = prices.rolling(window=window).mean()
        std_dev = prices.rolling(window=window).std()
        upper_band = middle_band + (std_dev * num_std)
        lower_band = middle_band - (std_dev * num_std)
        
        return middle_band, upper_band, lower_band
    
    def _calculate_macd(self, prices, fast_period=12, slow_period=26, signal_period=9):
        """
        Calculate MACD (Moving Average Convergence Divergence).
        
        Args:
            prices (pd.Series): Series of prices
            fast_period (int): Fast EMA period, default 12
            slow_period (int): Slow EMA period, default 26
            signal_period (int): Signal line period, default 9
            
        Returns:
            tuple: (macd_line, signal_line, histogram)
        """
        fast_ema = prices.ewm(span=fast_period, adjust=False).mean()
        slow_ema = prices.ewm(span=slow_period, adjust=False).mean()
        macd_line = fast_ema - slow_ema
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
        
    def calculate_technical_indicators(self, df):
        """
        Calculate various technical indicators for signal generation.
        
        Args:
            df (pd.DataFrame): DataFrame with OHLC data
            
        Returns:
            pd.DataFrame: DataFrame with added technical indicators
        """
        # Make a copy to avoid modifying the original
        df_with_indicators = df.copy()
        
        # Add RSI
        df_with_indicators['rsi'] = self._calculate_rsi(df['close'])
        
        # Add Bollinger Bands
        middle, upper, lower = self._calculate_bollinger_bands(df['close'])
        df_with_indicators['bollinger_mid'] = middle
        df_with_indicators['bollinger_upper'] = upper
        df_with_indicators['bollinger_lower'] = lower
        
        # Add MACD
        macd_line, signal_line, histogram = self._calculate_macd(df['close'])
        df_with_indicators['macd'] = macd_line
        df_with_indicators['macd_signal'] = signal_line
        df_with_indicators['macd_histogram'] = histogram
        
        # Add Ichimoku Cloud
        df_with_indicators = self._calculate_ichimoku(df_with_indicators)
        
        return df_with_indicators

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
