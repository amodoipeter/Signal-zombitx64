import numpy as np
import pandas as pd
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging

from app.core.market_data.fetcher import MarketDataFetcher
from app.core.ai.indicators import calculate_all_indicators
from app.core.ai.model_loader import load_model, ModelType
from app.models.signal import Signal, SignalType, SignalStatus

logger = logging.getLogger(__name__)

class AISignalGenerator:
    def __init__(self):
        """Initialize the AI signal generator."""
        self.data_fetcher = MarketDataFetcher()
        
        # Load the requested model type based on environment variable
        model_type_str = os.getenv("AI_MODEL_TYPE", "ensemble").lower()
        try:
            model_type = ModelType(model_type_str)
        except ValueError:
            logger.warning(f"Unknown model type: {model_type_str}, using ensemble model")
            model_type = ModelType.ENSEMBLE
            
        self.ai_model = load_model(model_type)
        logger.info(f"Initialized AI Signal Generator with {model_type} model")
        
        # Prediction mapping
        self.prediction_mapping = {
            0: SignalType.BUY,
            1: SignalType.SELL,
            2: None  # Hold - no signal
        }
        
        # Minimum confidence threshold
        self.min_confidence = float(os.getenv("SIGNAL_CONFIDENCE_THRESHOLD", 75.0))
    
    async def generate_signals(self, symbols: List[str], timeframes: List[str]) -> List[Signal]:
        """Generate trading signals for given symbols and timeframes."""
        signals = []
        
        for symbol in symbols:
            for timeframe in timeframes:
                try:
                    # 1. Fetch market data with enough history for indicators
                    market_data = await self.data_fetcher.fetch_data(symbol, timeframe, limit=300)
                    
                    if market_data.empty:
                        logger.warning(f"No data available for {symbol} on {timeframe}")
                        continue
                    
                    # 2. Calculate technical indicators
                    data_with_indicators = calculate_all_indicators(market_data)
                    
                    # 3. Prepare features for AI model
                    features = self._prepare_features(data_with_indicators)
                    
                    # 4. Get AI prediction
                    prediction_result = self._get_prediction(features, data_with_indicators)
                    
                    # 5. If prediction is valid and confidence is high enough, create a signal
                    if prediction_result:
                        signal_type, confidence, analysis, tp, sl = prediction_result
                        
                        if signal_type and confidence >= self.min_confidence:
                            # Determine market type
                            if '/USD' in symbol or 'USDT' in symbol:
                                market_type = 'crypto'
                            elif any(curr in symbol for curr in ['JPY', 'USD', 'EUR', 'GBP', 'AUD', 'CAD', 'CHF']):
                                market_type = 'forex'
                            else:
                                market_type = 'other'
                            
                            # Current price
                            current_price = market_data['close'].iloc[-1]
                            
                            # Create signal
                            signal = Signal(
                                symbol=symbol,
                                market=market_type,
                                signal_type=signal_type,
                                entry_price=current_price,
                                take_profit=tp,
                                stop_loss=sl,
                                risk_reward_ratio=abs((tp - current_price) / (current_price - sl)) if sl != current_price else 0,
                                timeframe=timeframe,
                                confidence_score=int(confidence),
                                analysis_summary=analysis,
                                ai_model_version="2.0",
                                strategy_name=self._determine_strategy(data_with_indicators),
                                strategy_category=self._determine_strategy_category(timeframe),
                                indicators_data=self._get_indicator_snapshot(data_with_indicators)
                            )
                            signals.append(signal)
                
                except Exception as e:
                    logger.error(f"Error generating signal for {symbol} {timeframe}: {str(e)}")
        
        # Close the data fetcher
        await self.data_fetcher.close()
        
        return signals
    
    def _prepare_features(self, data: pd.DataFrame) -> np.ndarray:
        """Prepare feature matrix for the AI model."""
        # Drop rows with NaN values
        data = data.dropna()
        
        # Select relevant features - these should match what the model was trained on
        feature_cols = [
            # Price data
            'open', 'high', 'low', 'close',
            
            # Technical indicators
            'rsi', 'macd', 'macd_signal', 'macd_hist',
            'bb_upper', 'bb_middle', 'bb_lower',
            'ichi_tenkan', 'ichi_kijun', 'ichi_senkou_a', 'ichi_senkou_b',
            'stoch_k', 'stoch_d', 
            'adx', 'plus_di', 'minus_di',
            'sma_20', 'sma_50', 'sma_200',
            'ema_9', 'ema_21',
            'atr'
        ]
        
        # Filter out only available columns
        available_cols = [col for col in feature_cols if col in data.columns]
        
        # Get the features as numpy array
        features = data[available_cols].values
        
        # Only use the last N periods for prediction
        lookback = min(60, features.shape[0])
        return features[-lookback:]
    
    def _get_prediction(self, features: np.ndarray, data: pd.DataFrame) -> Optional[Tuple]:
        """
        Get prediction from the AI model.
        
        Returns:
            Tuple of (signal_type, confidence, analysis, take_profit, stop_loss) or None if no signal
        """
        try:
            # Get prediction from model
            predictions = self.ai_model.predict(features)
            
            # Extract the most recent prediction
            prediction_class, confidence = predictions[-1]
            
            # Map prediction class to signal type
            signal_type = self.prediction_mapping.get(prediction_class)
            
            # If no signal (HOLD), return None
            if signal_type is None:
                return None
            
            # Get current price data
            current_price = data['close'].iloc[-1]
            atr = data['atr'].iloc[-1] if 'atr' in data else current_price * 0.01
            
            # Calculate take profit and stop loss based on ATR
            if signal_type == SignalType.BUY:
                take_profit = current_price + (atr * 3)  # 3x ATR for TP
                stop_loss = current_price - (atr * 1.5)  # 1.5x ATR for SL
                
                analysis = self._generate_buy_analysis(data)
            else:  # SELL signal
                take_profit = current_price - (atr * 3)  # 3x ATR for TP
                stop_loss = current_price + (atr * 1.5)  # 1.5x ATR for SL
                
                analysis = self._generate_sell_analysis(data)
            
            return signal_type, confidence, analysis, take_profit, stop_loss
            
        except Exception as e:
            logger.error(f"Error getting prediction: {str(e)}")
            return None
    
    def _generate_buy_analysis(self, data: pd.DataFrame) -> str:
        """Generate analysis text for a BUY signal."""
        analysis_points = []
        
        # Check RSI
        if 'rsi' in data:
            rsi = data['rsi'].iloc[-1]
            if rsi < 30:
                analysis_points.append(f"RSI is oversold at {rsi:.1f}, suggesting a potential reversal to the upside")
            elif rsi < 50:
                analysis_points.append(f"RSI is rising from low levels at {rsi:.1f}, showing bullish momentum building")
            else:
                analysis_points.append(f"RSI at {rsi:.1f} shows strong bullish momentum")
        
        # Check MACD
        if 'macd' in data and 'macd_signal' in data:
            macd = data['macd'].iloc[-1]
            signal = data['macd_signal'].iloc[-1]
            if macd > signal and macd - signal < 0.001:
                analysis_points.append("MACD just crossed above signal line, suggesting start of bullish move")
            elif macd > signal:
                analysis_points.append("MACD above signal line confirms bullish momentum")
        
        # Check Bollinger Bands
        if 'bb_lower' in data:
            close = data['close'].iloc[-1]
            bb_lower = data['bb_lower'].iloc[-1]
            if close < bb_lower:
                analysis_points.append("Price broke below lower Bollinger Band, suggesting oversold condition ready for reversal")
        
        # Check Ichimoku Cloud
        if 'ichi_above_cloud' in data:
            above_cloud = data['ichi_above_cloud'].iloc[-1]
            if above_cloud:
                analysis_points.append("Price is above Ichimoku Cloud, confirming bullish trend")
        
        # Moving Averages
        if 'sma_20' in data and 'sma_50' in data:
            sma_20 = data['sma_20'].iloc[-1]
            sma_50 = data['sma_50'].iloc[-1]
            if sma_20 > sma_50 and sma_20 / sma_50 < 1.01:
                analysis_points.append("20 SMA just crossed above 50 SMA, signaling a bullish trend change")
            elif sma_20 > sma_50:
                analysis_points.append("20 SMA above 50 SMA confirms bullish trend")
        
        # If no specific conditions met, provide general analysis
        if not analysis_points:
            analysis_points.append("Multiple technical indicators suggest a bullish setup with favorable risk-reward")
        
        # Add risk warning
        analysis_points.append("Always use proper risk management and consider current market volatility")
        
        return ". ".join(analysis_points) + "."
    
    def _generate_sell_analysis(self, data: pd.DataFrame) -> str:
        """Generate analysis text for a SELL signal."""
        analysis_points = []
        
        # Check RSI
        if 'rsi' in data:
            rsi = data['rsi'].iloc[-1]
            if rsi > 70:
                analysis_points.append(f"RSI is overbought at {rsi:.1f}, suggesting a potential reversal to the downside")
            elif rsi > 50:
                analysis_points.append(f"RSI is falling from high levels at {rsi:.1f}, showing bearish momentum building")
            else:
                analysis_points.append(f"RSI at {rsi:.1f} shows strong bearish momentum")
        
        # Check MACD
        if 'macd' in data and 'macd_signal' in data:
            macd = data['macd'].iloc[-1]
            signal = data['macd_signal'].iloc[-1]
            if macd < signal and signal - macd < 0.001:
                analysis_points.append("MACD just crossed below signal line, suggesting start of bearish move")
            elif macd < signal:
                analysis_points.append("MACD below signal line confirms bearish momentum")
        
        # Check Bollinger Bands
        if 'bb_upper' in data:
            close = data['close'].iloc[-1]
            bb_upper = data['bb_upper'].iloc[-1]
            if close > bb_upper:
                analysis_points.append("Price broke above upper Bollinger Band, suggesting overbought condition ready for reversal")
        
        # Check Ichimoku Cloud
        if 'ichi_below_cloud' in data:
            below_cloud = data['ichi_below_cloud'].iloc[-1]
            if below_cloud:
                analysis_points.append("Price is below Ichimoku Cloud, confirming bearish trend")
        
        # Moving Averages
        if 'sma_20' in data and 'sma_50' in data:
            sma_20 = data['sma_20'].iloc[-1]
            sma_50 = data['sma_50'].iloc[-1]
            if sma_20 < sma_50 and sma_50 / sma_20 < 1.01:
                analysis_points.append("20 SMA just crossed below 50 SMA, signaling a bearish trend change")
            elif sma_20 < sma_50:
                analysis_points.append("20 SMA below 50 SMA confirms bearish trend")
        
        # If no specific conditions met, provide general analysis
        if not analysis_points:
            analysis_points.append("Multiple technical indicators suggest a bearish setup with favorable risk-reward")
        
        # Add risk warning
        analysis_points.append("Always use proper risk management and consider current market volatility")
        
        return ". ".join(analysis_points) + "."
    
    def _determine_strategy(self, data: pd.DataFrame) -> str:
        """Determine the strategy name based on indicator values."""
        # Check for oversold/overbought RSI
        if 'rsi' in data:
            rsi = data['rsi'].iloc[-1]
            if rsi < 30:
                return "RSI Oversold Reversal"
            elif rsi > 70:
                return "RSI Overbought Reversal"
        
        # Check for MACD crossover
        if 'macd' in data and 'macd_signal' in data:
            macd = data['macd'].iloc[-2:].values
            signal = data['macd_signal'].iloc[-2:].values
            if macd[0] < signal[0] and macd[1] > signal[1]:
                return "MACD Bullish Crossover"
            elif macd[0] > signal[0] and macd[1] < signal[1]:
                return "MACD Bearish Crossover"
        
        # Check for Bollinger Band squeeze
        if 'bb_upper' in data and 'bb_lower' in data:
            bb_upper = data['bb_upper'].iloc[-1]
            bb_lower = data['bb_lower'].iloc[-1]
            close = data['close'].iloc[-1]
            bb_width = (bb_upper - bb_lower) / close
            if bb_width < 0.03:
                return "Bollinger Band Squeeze"
        
        # Check for Ichimoku signals
        if 'ichi_tenkan' in data and 'ichi_kijun' in data:
            tenkan = data['ichi_tenkan'].iloc[-2:].values
            kijun = data['ichi_kijun'].iloc[-2:].values
            if tenkan[0] < kijun[0] and tenkan[1] > kijun[1]:
                return "Ichimoku TK Cross Bullish"
            elif tenkan[0] > kijun[0] and tenkan[1] < kijun[1]:
                return "Ichimoku TK Cross Bearish"
        
        # Check for trend following with ADX
        if 'adx' in data:
            adx = data['adx'].iloc[-1]
            if adx > 25:
                if 'plus_di' in data and 'minus_di' in data:
                    plus_di = data['plus_di'].iloc[-1]
                    minus_di = data['minus_di'].iloc[-1]
                    if plus_di > minus_di:
                        return "Strong Trend Following Bullish"
                    else:
                        return "Strong Trend Following Bearish"
        
        # Default to ensemble strategy
        return "AI Ensemble Strategy"
    
    def _determine_strategy_category(self, timeframe: str) -> str:
        """Determine strategy category based on timeframe."""
        if timeframe in ['1m', '3m', '5m', '15m']:
            return "Scalping"
        elif timeframe in ['30m', '1h', '2h', '4h']:
            return "Swing"
        else:  # '6h', '12h', '1d', '3d', '1w'
            return "Position"
    
    def _get_indicator_snapshot(self, data: pd.DataFrame) -> Dict:
        """Get snapshot of important indicators for storing with signal."""
        indicators = {}
        
        # Key indicators to save
        key_indicators = [
            'rsi', 'macd', 'macd_signal', 'adx', 'atr',
            'bb_upper', 'bb_lower', 'bb_middle',
            'ichi_tenkan', 'ichi_kijun', 'stoch_k', 'stoch_d'
        ]
        
        # Add available indicators to snapshot
        for indicator in key_indicators:
            if indicator in data.columns:
                indicators[indicator] = float(data[indicator].iloc[-1])
        
        # Add some derived values
        if 'bb_upper' in indicators and 'bb_lower' in indicators:
            indicators['bb_width'] = indicators['bb_upper'] - indicators['bb_lower']
        
        if 'close' in data.columns:
            indicators['close'] = float(data['close'].iloc[-1])
            
            # Add distance from key MAs if available
            for ma in ['sma_20', 'sma_50', 'sma_200']:
                if ma in data.columns:
                    indicators[f'{ma}_dist_pct'] = (
                        (float(data['close'].iloc[-1]) / float(data[ma].iloc[-1]) - 1) * 100
                    )
        
        return indicators
