import pandas as pd
import numpy as np
import talib
from typing import Dict

def calculate_rsi(data: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Relative Strength Index."""
    return talib.RSI(data['close'], timeperiod=period)

def calculate_macd(data: pd.DataFrame) -> Dict[str, pd.Series]:
    """Calculate MACD, Signal Line and Histogram."""
    macd, signal, hist = talib.MACD(
        data['close'],
        fastperiod=12,
        slowperiod=26,
        signalperiod=9
    )
    return {
        'macd': macd,
        'signal': signal,
        'hist': hist
    }

def calculate_bollinger_bands(data: pd.DataFrame, period: int = 20, stdev: float = 2.0) -> Dict[str, pd.Series]:
    """Calculate Bollinger Bands."""
    upper, middle, lower = talib.BBANDS(
        data['close'],
        timeperiod=period,
        nbdevup=stdev,
        nbdevdn=stdev
    )
    return {
        'upper': upper,
        'middle': middle,
        'lower': lower
    }

def calculate_ichimoku(data: pd.DataFrame) -> Dict[str, pd.Series]:
    """Calculate Ichimoku Cloud components."""
    high = data['high']
    low = data['low']
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = high.rolling(window=9).max()
    period9_low = low.rolling(window=9).min()
    tenkan_sen = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = high.rolling(window=26).max()
    period26_low = low.rolling(window=26).min()
    kijun_sen = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Conversion Line + Base Line)/2
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2
    period52_high = high.rolling(window=52).max()
    period52_low = low.rolling(window=52).min()
    senkou_span_b = ((period52_high + period52_low) / 2).shift(26)
    
    # Chikou Span (Lagging Span): Close price shifted -26 periods
    chikou_span = data['close'].shift(-26)
    
    return {
        'tenkan_sen': tenkan_sen,
        'kijun_sen': kijun_sen,
        'senkou_span_a': senkou_span_a,
        'senkou_span_b': senkou_span_b,
        'chikou_span': chikou_span
    }

def calculate_atr(data: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Average True Range."""
    return talib.ATR(data['high'], data['low'], data['close'], timeperiod=period)

def calculate_all_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate all technical indicators and add them to the dataframe."""
    df = data.copy()
    
    # RSI
    df['rsi'] = calculate_rsi(df)
    
    # MACD
    macd_values = calculate_macd(df)
    df['macd'] = macd_values['macd']
    df['macd_signal'] = macd_values['signal']
    df['macd_hist'] = macd_values['hist']
    
    # Bollinger Bands
    bb_values = calculate_bollinger_bands(df)
    df['bb_upper'] = bb_values['upper']
    df['bb_middle'] = bb_values['middle']
    df['bb_lower'] = bb_values['lower']
    
    # Ichimoku Cloud
    ichimoku_values = calculate_ichimoku(df)
    df['ichi_tenkan'] = ichimoku_values['tenkan_sen']
    df['ichi_kijun'] = ichimoku_values['kijun_sen']
    df['ichi_senkou_a'] = ichimoku_values['senkou_span_a']
    df['ichi_senkou_b'] = ichimoku_values['senkou_span_b']
    df['ichi_chikou'] = ichimoku_values['chikou_span']
    
    # ATR
    df['atr'] = calculate_atr(df)
    
    return df
