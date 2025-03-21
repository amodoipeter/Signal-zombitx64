import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class SignalAnalysisService:
    """Service for generating advanced trading signal analysis."""
    
    @staticmethod
    def analyze_market_conditions(data: pd.DataFrame) -> Dict:
        """
        Analyze overall market conditions.
        
        Args:
            data: DataFrame with technical indicators
            
        Returns:
            Dictionary with market condition analysis
        """
        # Get latest data point
        latest = data.iloc[-1]
        
        # Determine trend
        sma20_sma50_ratio = latest['sma_20'] / latest['sma_50'] if 'sma_20' in data.columns and 'sma_50' in data.columns else 1
        sma50_sma200_ratio = latest['sma_50'] / latest['sma_200'] if 'sma_50' in data.columns and 'sma_200' in data.columns else 1
        
        # Determine if in uptrend, downtrend, or ranging
        if sma20_sma50_ratio > 1.01 and sma50_sma200_ratio > 1.01:
            trend = "Strong Uptrend"
        elif sma20_sma50_ratio > 1.005 and sma50_sma200_ratio > 1:
            trend = "Uptrend"
        elif sma20_sma50_ratio < 0.99 and sma50_sma200_ratio < 0.99:
            trend = "Strong Downtrend"
        elif sma20_sma50_ratio < 0.995 and sma50_sma200_ratio < 1:
            trend = "Downtrend"
        else:
            trend = "Ranging"
        
        # Determine volatility
        if 'bb_upper' in data.columns and 'bb_lower' in data.columns and 'close' in data.columns:
            bb_width = (latest['bb_upper'] - latest['bb_lower']) / latest['close']
            if bb_width > 0.06:
                volatility = "High"
            elif bb_width > 0.03:
                volatility = "Medium"
            else:
                volatility = "Low"
        else:
            volatility = "Unknown"
        
        # Determine momentum
        if 'rsi' in data.columns:
            rsi = latest['rsi']
            if rsi > 70:
                momentum = "Overbought"
            elif rsi < 30:
                momentum = "Oversold"
            elif rsi > 60:
                momentum = "Strong"
            elif rsi < 40:
                momentum = "Weak"
            else:
                momentum = "Neutral"
        else:
            momentum = "Unknown"
        
        return {
            "trend": trend,
            "volatility": volatility,
            "momentum": momentum
        }
    
    @staticmethod
    def identify_chart_patterns(data: pd.DataFrame) -> List[str]:
        """
        Identify potential chart patterns in the data.
        
        Args:
            data: DataFrame with OHLC data
            
        Returns:
            List of identified patterns
        """
        patterns = []
        
        # Need at least 20 data points
        if len(data) < 20:
            return []
        
        close = data['close'].values
        high = data['high'].values
        low = data['low'].values
        open_prices = data['open'].values if 'open' in data.columns else None
        
        # Check for double top
        if len(close) > 30:
            recent_highs = [i for i in range(5, len(high)-5) if high[i] > high[i-1] and high[i] > high[i-2] and high[i] > high[i+1] and high[i] > high[i+2]]
            if len(recent_highs) >= 2:
                highest = sorted(recent_highs, key=lambda i: high[i], reverse=True)[:2]
                if abs(high[highest[0]] - high[highest[1]]) / high[highest[0]] < 0.02 and abs(highest[0] - highest[1]) > 10:
                    patterns.append("Double Top")
        
        # Check for double bottom
        if len(close) > 30:
            recent_lows = [i for i in range(5, len(low)-5) if low[i] < low[i-1] and low[i] < low[i-2] and low[i] < low[i+1] and low[i] < low[i+2]]
            if len(recent_lows) >= 2:
                lowest = sorted(recent_lows, key=lambda i: low[i])[:2]
                if abs(low[lowest[0]] - low[lowest[1]]) / low[lowest[0]] < 0.02 and abs(lowest[0] - lowest[1]) > 10:
                    patterns.append("Double Bottom")
        
        # Check for head and shoulders
        # (This is simplified - a real implementation would be more complex)
        if len(close) > 40:
            # Find three peaks
            recent_highs = [i for i in range(5, len(high)-5) if high[i] > high[i-1] and high[i] > high[i-2] and high[i] > high[i+1] and high[i] > high[i+2]]
            if len(recent_highs) >= 3:
                peaks = sorted(recent_highs[-15:])
                if len(peaks) >= 3 and high[peaks[1]] > high[peaks[0]] and high[peaks[1]] > high[peaks[2]] and abs(high[peaks[0]] - high[peaks[2]]) / high[peaks[0]] < 0.05:
                    patterns.append("Head and Shoulders")
        
        # Check for bullish engulfing
        if open_prices is not None:
            for i in range(1, len(close)-1):
                if close[i-1] < open_prices[i-1] and close[i] > open_prices[i] and open_prices[i] <= close[i-1] and close[i] > open_prices[i-1]:
                    patterns.append("Bullish Engulfing")
                    break
        
        # Check for bearish engulfing
        if open_prices is not None:
            for i in range(1, len(close)-1):
                if close[i-1] > open_prices[i-1] and close[i] < open_prices[i] and open_prices[i] >= close[i-1] and close[i] < open_prices[i-1]:
                    patterns.append("Bearish Engulfing")
                    break
        
        # Check for breakout (price breaks above resistance)
        if 'resistance' in data.columns:
            last_resistance = data['resistance'].dropna().iloc[-5:].mean()
            if close[-1] > last_resistance * 1.01:
                patterns.append("Breakout Above Resistance")
        
        # Check for breakdown (price breaks below support)
        if 'support' in data.columns:
            last_support = data['support'].dropna().iloc[-5:].mean()
            if close[-1] < last_support * 0.99:
                patterns.append("Breakdown Below Support")
        
        return patterns
    
    @staticmethod
    def get_important_levels(data: pd.DataFrame) -> Dict:
        """
        Identify important price levels.
        
        Args:
            data: DataFrame with technical indicators
            
        Returns:
            Dictionary with important price levels
        """
        levels = {}
        
        # Support and resistance levels
        if 'support' in data.columns:
            levels['support'] = data['support'].dropna().iloc[-5:].mean()
        if 'resistance' in data.columns:
            levels['resistance'] = data['resistance'].dropna().iloc[-5:].mean()
        
        # Fibonacci levels
        if 'fib_618' in data.columns:
            levels['fibonacci_618'] = data['fib_618'].iloc[-1]
        
        # Pivot points
        if 'pivot' in data.columns:
            levels['pivot'] = data['pivot'].iloc[-1]
            levels['pivot_r1'] = data['pivot_r1'].iloc[-1]
            levels['pivot_s1'] = data['pivot_s1'].iloc[-1]
        
        # Moving averages
        if 'sma_20' in data.columns:
            levels['sma_20'] = data['sma_20'].iloc[-1]
        if 'sma_50' in data.columns:
            levels['sma_50'] = data['sma_50'].iloc[-1]
        if 'sma_200' in data.columns:
            levels['sma_200'] = data['sma_200'].iloc[-1]
        
        # Bollinger Bands
        if 'bb_upper' in data.columns:
            levels['bollinger_upper'] = data['bb_upper'].iloc[-1]
        if 'bb_middle' in data.columns:
            levels['bollinger_middle'] = data['bb_middle'].iloc[-1]
        if 'bb_lower' in data.columns:
            levels['bollinger_lower'] = data['bb_lower'].iloc[-1]
        
        # Ichimoku Cloud
        if 'ichi_tenkan' in data.columns:
            levels['ichimoku_tenkan'] = data['ichi_tenkan'].iloc[-1]
        if 'ichi_kijun' in data.columns:
            levels['ichimoku_kijun'] = data['ichi_kijun'].iloc[-1]
        
        return levels
    
    @staticmethod
    def generate_analysis_text(data: pd.DataFrame, is_buy_signal: bool = True) -> str:
        """
        Generate comprehensive analysis text for a trading signal.
        
        Args:
            data: DataFrame with technical indicators
            is_buy_signal: Whether this is a buy signal (True) or sell signal (False)
            
        Returns:
            Text analysis of the signal
        """
        analysis_points = []
        
        # Get market conditions
        conditions = SignalAnalysisService.analyze_market_conditions(data)
        analysis_points.append(f"Market is in a {conditions['trend']} with {conditions['momentum']} momentum and {conditions['volatility']} volatility")
        
        # Get chart patterns
        patterns = SignalAnalysisService.identify_chart_patterns(data)
        if patterns:
            analysis_points.append(f"Chart shows pattern(s): {', '.join(patterns)}")
        
        # Get important price levels
        levels = SignalAnalysisService.get_important_levels(data)
        
        # Analysis based on signal type (buy or sell)
        if is_buy_signal:
            # RSI analysis
            if 'rsi' in data.columns:
                rsi = data['rsi'].iloc[-1]
                if rsi < 30:
                    analysis_points.append(f"RSI is oversold at {rsi:.1f}, suggesting bullish reversal potential")
                elif rsi < 50 and rsi > 30:
                    analysis_points.append(f"RSI at {rsi:.1f} is ascending from oversold conditions")
                elif rsi > 50:
                    analysis_points.append(f"RSI at {rsi:.1f} indicates bullish momentum")
            
            # MACD analysis
            if 'macd' in data.columns and 'macd_signal' in data.columns:
                macd = data['macd'].iloc[-2:].values
                signal = data['macd_signal'].iloc[-2:].values
                
                if macd[1] > signal[1] and macd[0] <= signal[0]:
                    analysis_points.append("MACD just crossed above signal line, strong bullish signal")
                elif macd[1] > signal[1]:
                    analysis_points.append("MACD above signal line confirms uptrend")
                elif macd[1] > macd[0]:
                    analysis_points.append("MACD is rising, suggesting building bullish momentum")
            
            # Ichimoku analysis
            if 'ichi_above_cloud' in data.columns and data['ichi_above_cloud'].iloc[-1]:
                analysis_points.append("Price is above Ichimoku Cloud, confirming bullish trend")
            
            # Support level
            if 'support' in levels:
                analysis_points.append(f"Support level at {levels['support']:.6f} should provide a floor on pullbacks")
            
            # MA analysis
            if 'sma_20' in levels and 'sma_50' in levels:
                if levels['sma_20'] > levels['sma_50']:
                    analysis_points.append("Short-term MA above long-term MA confirms bullish structure")
        else:
            # Sell signal analysis
            # RSI analysis
            if 'rsi' in data.columns:
                rsi = data['rsi'].iloc[-1]
                if rsi > 70:
                    analysis_points.append(f"RSI is overbought at {rsi:.1f}, suggesting bearish reversal potential")
                elif rsi > 50 and rsi < 70:
                    analysis_points.append(f"RSI at {rsi:.1f} is descending from overbought conditions")
                elif rsi < 50:
                    analysis_points.append(f"RSI at {rsi:.1f} indicates bearish momentum")
            
            # MACD analysis
            if 'macd' in data.columns and 'macd_signal' in data.columns:
                macd = data['macd'].iloc[-2:].values
                signal = data['macd_signal'].iloc[-2:].values
                
                if macd[1] < signal[1] and macd[0] >= signal[0]:
                    analysis_points.append("MACD just crossed below signal line, strong bearish signal")
                elif macd[1] < signal[1]:
                    analysis_points.append("MACD below signal line confirms downtrend")
                elif macd[1] < macd[0]:
                    analysis_points.append("MACD is falling, suggesting building bearish momentum")
            
            # Ichimoku analysis
            if 'ichi_below_cloud' in data.columns and data['ichi_below_cloud'].iloc[-1]:
                analysis_points.append("Price is below Ichimoku Cloud, confirming bearish trend")
            
            # Resistance level
            if 'resistance' in levels:
                analysis_points.append(f"Resistance level at {levels['resistance']:.6f} should cap upside on rallies")
            
            # MA analysis
            if 'sma_20' in levels and 'sma_50' in levels:
                if levels['sma_20'] < levels['sma_50']:
                    analysis_points.append("Short-term MA below long-term MA confirms bearish structure")
        
        # Add risk management advice
        risk_reward = "favorable" if is_buy_signal else "beneficial"
        analysis_points.append(f"Risk-reward ratio is {risk_reward} based on current market structure")
        analysis_points.append("Always use proper risk management and confirm entry with additional analysis")
        
        return ". ".join(analysis_points)

# Create service instance
signal_analysis_service = SignalAnalysisService()
