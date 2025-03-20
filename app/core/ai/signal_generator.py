import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging

from app.core.market_data.fetcher import MarketDataFetcher
from app.core.ai.indicators import calculate_all_indicators
from app.core.ai.model_loader import load_model
from app.models.signal import Signal, SignalType, SignalStatus

logger = logging.getLogger(__name__)

class AISignalGenerator:
    def __init__(self):
        self.data_fetcher = MarketDataFetcher()
        self.ai_model = load_model()
    
    async def generate_signals(self, symbols: List[str], timeframes: List[str]) -> List[Signal]:
        """Generate trading signals for given symbols and timeframes."""
        signals = []
        
        for symbol in symbols:
            for timeframe in timeframes:
                try:
                    # 1. Fetch market data
                    market_data = await self.data_fetcher.fetch_data(symbol, timeframe)
                    
                    if market_data.empty:
                        logger.warning(f"No data available for {symbol} on {timeframe}")
                        continue
                    
                    # 2. Calculate technical indicators
                    data_with_indicators = calculate_all_indicators(market_data)
                    
                    # 3. Prepare features for AI model
                    features = self._prepare_features(data_with_indicators)
                    
                    # 4. Get AI prediction
                    prediction, confidence, analysis = self._get_prediction(features)
                    
                    # 5. If confidence is high enough, create a signal
                    if confidence >= 70:  # Only create signals with high confidence
                        signal = self._create_signal(
                            symbol=symbol,
                            timeframe=timeframe,
                            prediction=prediction,
                            confidence=confidence,
                            analysis=analysis,
                            data=data_with_indicators
                        )
                        signals.append(signal)
                
                except Exception as e:
                    logger.error(f"Error generating signal for {symbol} {timeframe}: {str(e)}")
        
        return signals
    
    def _prepare_features(self, data: pd.DataFrame) -> np.ndarray:
        """Prepare feature matrix for the AI model."""
        # Drop unnecessary columns and handle NaN values
        feature_cols = [
            'open', 'high', 'low', 'close', 'volume',
            'rsi', 'macd', 'macd_signal', 'macd_hist',
            'bb_upper', 'bb_middle', 'bb_lower',
            'ichi_tenkan', 'ichi_kijun', 'ichi_senkou_a', 'ichi_senkou_b'
        ]
        
        features = data[feature_cols].fillna(0).values
        return features
    
    def _get_prediction(self, features: np.ndarray) -> Tuple[SignalType, int, str]:
        """Get AI prediction for signal type, confidence score, and analysis."""
        # In a real implementation, this would call the ML model
        # For now, using a simple rule-based example
        
        latest_features = features[-1]
        
        # Simple rule based on RSI for demonstration
        rsi = latest_features[6]  # Assuming RSI is at index 6
        
        if rsi < 30:
            signal_type = SignalType.BUY
            confidence = 85
            analysis = "RSI indicates oversold conditions. Price could reverse upward soon."
        elif rsi > 70:
            signal_type = SignalType.SELL
            confidence = 82
            analysis = "RSI indicates overbought conditions. Price could reverse downward soon."
        else:
            # If no strong signal, return with lower confidence
            if np.random.random() > 0.5:  # Just for demonstration
                signal_type = SignalType.BUY
            else:
                signal_type = SignalType.SELL
                
            confidence = 50
            analysis = "Market conditions are neutral. No strong signal detected."
        
        return signal_type, confidence, analysis
    
    def _create_signal(self, symbol, timeframe, prediction, confidence, analysis, data) -> Signal:
        """Create a Signal object based on AI prediction."""
        current_price = data['close'].iloc[-1]
        
        # Calculate take profit and stop loss based on volatility
        atr = data['atr'].iloc[-1] if 'atr' in data.columns else current_price * 0.01
        
        if prediction == SignalType.BUY:
            take_profit = current_price * 1.02  # 2% profit target
            stop_loss = current_price - (atr * 1.5)  # 1.5x ATR for stop loss
        else:  # SELL
            take_profit = current_price * 0.98  # 2% profit target
            stop_loss = current_price + (atr * 1.5)  # 1.5x ATR for stop loss
        
        # Calculate risk-reward ratio
        if prediction == SignalType.BUY:
            risk = current_price - stop_loss
            reward = take_profit - current_price
        else:  # SELL
            risk = stop_loss - current_price
            reward = current_price - take_profit
            
        risk_reward_ratio = reward / risk if risk > 0 else 0
        
        # Detect trading strategy category
        if timeframe in ['1m', '5m', '15m']:
            strategy_category = "Scalping"
        elif timeframe in ['1h', '4h']:
            strategy_category = "Swing"
        else:
            strategy_category = "Trend"
        
        return Signal(
            symbol=symbol,
            market="crypto" if "USD" in symbol else "forex",
            signal_type=prediction,
            entry_price=current_price,
            take_profit=take_profit,
            stop_loss=stop_loss,
            risk_reward_ratio=risk_reward_ratio,
            timeframe=timeframe,
            confidence_score=confidence,
            analysis_summary=analysis,
            ai_model_version="1.0.0",
            strategy_name="AI Adaptive Strategy",
            strategy_category=strategy_category,
            indicators_data={
                "rsi": float(data['rsi'].iloc[-1]) if 'rsi' in data.columns else None,
                "macd": float(data['macd'].iloc[-1]) if 'macd' in data.columns else None
            }
        )
