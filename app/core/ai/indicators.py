import pandas as pd
import numpy as np
import talib
from typing import Dict, Optional, List, Tuple
import math

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
    """
    Calculate complete Ichimoku Cloud components.
    
    Components:
    - Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    - Kijun-sen (Base Line): (26-period high + 26-period low)/2
    - Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2
    - Senkou Span B (Leading Span B): (52-period high + 52-period low)/2
    - Chikou Span (Lagging Span): Close price shifted -26 periods
    """
    high = data['high']
    low = data['low']
    close = data['close']
    
    # Tenkan-sen (Conversion Line)
    period9_high = high.rolling(window=9).max()
    period9_low = low.rolling(window=9).min()
    tenkan_sen = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line)
    period26_high = high.rolling(window=26).max()
    period26_low = low.rolling(window=26).min()
    kijun_sen = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A)
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)
    
    # Senkou Span B (Leading Span B)
    period52_high = high.rolling(window=52).max()
    period52_low = low.rolling(window=52).min()
    senkou_span_b = ((period52_high + period52_low) / 2).shift(26)
    
    # Chikou Span (Lagging Span)
    chikou_span = close.shift(-26)
    
    # Cloud color (green when A > B, red when B > A)
    cloud_green = senkou_span_a > senkou_span_b
    cloud_red = senkou_span_b > senkou_span_a
    
    # Additional signals
    # Tenkan-sen crossing Kijun-sen (bullish when Tenkan > Kijun)
    tenkan_cross_kijun = tenkan_sen > kijun_sen
    
    # Price in relation to the cloud
    price_above_cloud = (close > senkou_span_a) & (close > senkou_span_b)
    price_below_cloud = (close < senkou_span_a) & (close < senkou_span_b)
    price_in_cloud = ~(price_above_cloud | price_below_cloud)
    
    return {
        'tenkan_sen': tenkan_sen,
        'kijun_sen': kijun_sen,
        'senkou_span_a': senkou_span_a,
        'senkou_span_b': senkou_span_b,
        'chikou_span': chikou_span,
        'cloud_green': cloud_green,
        'cloud_red': cloud_red,
        'tenkan_cross_kijun': tenkan_cross_kijun,
        'price_above_cloud': price_above_cloud,
        'price_in_cloud': price_in_cloud,
        'price_below_cloud': price_below_cloud
    }

def calculate_fibonacci_retracement(data: pd.DataFrame, window: int = 100) -> Dict[str, pd.Series]:
    """
    Calculate Fibonacci retracement levels for both uptrend and downtrend.
    
    Fibonacci levels: 0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0
    
    Returns:
        Dictionary containing Fibonacci levels for both uptrend and downtrend
    """
    high = data['high'].rolling(window=window).max()
    low = data['low'].rolling(window=window).min()
    
    # Uptrend: From low to high
    range_up = high - low
    level_0 = low  # 0.0
    level_236 = low + 0.236 * range_up  # 23.6%
    level_382 = low + 0.382 * range_up  # 38.2%
    level_500 = low + 0.5 * range_up    # 50.0%
    level_618 = low + 0.618 * range_up  # 61.8%
    level_786 = low + 0.786 * range_up  # 78.6%
    level_1000 = high  # 100.0%
    
    # Downtrend: From high to low
    range_down = high - low
    d_level_0 = high  # 0.0
    d_level_236 = high - 0.236 * range_down  # 23.6%
    d_level_382 = high - 0.382 * range_down  # 38.2%
    d_level_500 = high - 0.5 * range_down    # 50.0%
    d_level_618 = high - 0.618 * range_down  # 61.8%
    d_level_786 = high - 0.786 * range_down  # 78.6%
    d_level_1000 = low  # 100.0%
    
    return {
        'uptrend_0': level_0,
        'uptrend_236': level_236,
        'uptrend_382': level_382,
        'uptrend_500': level_500,
        'uptrend_618': level_618,
        'uptrend_786': level_786,
        'uptrend_1000': level_1000,
        'downtrend_0': d_level_0,
        'downtrend_236': d_level_236,
        'downtrend_382': d_level_382,
        'downtrend_500': d_level_500,
        'downtrend_618': d_level_618,
        'downtrend_786': d_level_786,
        'downtrend_1000': d_level_1000
    }

def calculate_pivot_points(data: pd.DataFrame, method: str = 'standard') -> Dict[str, pd.Series]:
    """
    Calculate pivot points using different methods.
    
    Args:
        data: DataFrame with OHLC data
        method: 'standard', 'fibonacci', 'camarilla', or 'woodie'
        
    Returns:
        Dictionary with pivot point levels
    """
    high = data['high'].shift(1)
    low = data['low'].shift(1)
    close = data['close'].shift(1)
    
    if method == 'standard':
        pivot = (high + low + close) / 3
        r1 = 2 * pivot - low
        r2 = pivot + (high - low)
        r3 = high + 2 * (pivot - low)
        s1 = 2 * pivot - high
        s2 = pivot - (high - low)
        s3 = low - 2 * (high - pivot)
        
    elif method == 'fibonacci':
        pivot = (high + low + close) / 3
        r1 = pivot + 0.382 * (high - low)
        r2 = pivot + 0.618 * (high - low)
        r3 = pivot + 1.000 * (high - low)
        s1 = pivot - 0.382 * (high - low)
        s2 = pivot - 0.618 * (high - low)
        s3 = pivot - 1.000 * (high - low)
        
    elif method == 'camarilla':
        pivot = (high + low + close) / 3
        r1 = close + 1.1 * (high - low) / 12
        r2 = close + 1.1 * (high - low) / 6
        r3 = close + 1.1 * (high - low) / 4
        s1 = close - 1.1 * (high - low) / 12
        s2 = close - 1.1 * (high - low) / 6
        s3 = close - 1.1 * (high - low) / 4
        
    elif method == 'woodie':
        pivot = (high + low + 2 * close) / 4
        r1 = 2 * pivot - low
        r2 = pivot + (high - low)
        s1 = 2 * pivot - high
        s2 = pivot - (high - low)
        r3 = high + 2 * (pivot - low)
        s3 = low - 2 * (high - pivot)
    
    else:
        raise ValueError(f"Unknown pivot point method: {method}")
    
    return {
        'pivot': pivot,
        'r1': r1,
        'r2': r2,
        'r3': r3,
        's1': s1,
        's2': s2,
        's3': s3
    }

def calculate_stochastic(data: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> Dict[str, pd.Series]:
    """Calculate Stochastic oscillator."""
    slowk, slowd = talib.STOCH(
        data['high'],
        data['low'],
        data['close'],
        fastk_period=k_period,
        slowk_period=3,
        slowk_matype=0,
        slowd_period=d_period,
        slowd_matype=0
    )
    return {
        'slowk': slowk,
        'slowd': slowd
    }

def calculate_adx(data: pd.DataFrame, period: int = 14) -> Dict[str, pd.Series]:
    """Calculate Average Directional Index (ADX), +DI and -DI."""
    adx = talib.ADX(data['high'], data['low'], data['close'], timeperiod=period)
    plus_di = talib.PLUS_DI(data['high'], data['low'], data['close'], timeperiod=period)
    minus_di = talib.MINUS_DI(data['high'], data['low'], data['close'], timeperiod=period)
    
    return {
        'adx': adx,
        'plus_di': plus_di,
        'minus_di': minus_di
    }

def calculate_obv(data: pd.DataFrame) -> pd.Series:
    """Calculate On-Balance Volume."""
    return talib.OBV(data['close'], data['volume'])

def calculate_atr(data: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Average True Range."""
    return talib.ATR(data['high'], data['low'], data['close'], timeperiod=period)

def calculate_support_resistance(data: pd.DataFrame, window: int = 20, threshold: float = 0.01) -> Dict[str, pd.Series]:
    """
    Calculate dynamic support and resistance levels using local minima and maxima.
    
    Args:
        data: DataFrame with OHLC data
        window: Window size to look for local min/max
        threshold: Minimum difference as percentage to consider a level
        
    Returns:
        Dictionary with support and resistance levels
    """
    close = data['close']
    high = data['high']
    low = data['low']
    
    resistance_levels = pd.Series(index=data.index, dtype='float64')
    support_levels = pd.Series(index=data.index, dtype='float64')
    
    for i in range(window, len(data)):
        # Get window of data
        window_high = high[i-window:i]
        window_low = low[i-window:i]
        current_close = close[i]
        
        # Find highest high and lowest low in window
        window_max = window_high.max()
        window_min = window_low.min()
        
        # Set resistance level
        if window_max > current_close * (1 + threshold):
            resistance_levels[i] = window_max
        else:
            resistance_levels[i] = np.nan
            
        # Set support level
        if window_min < current_close * (1 - threshold):
            support_levels[i] = window_min
        else:
            support_levels[i] = np.nan
    
    return {
        'support': support_levels,
        'resistance': resistance_levels
    }

def calculate_all_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate all technical indicators and add them to the dataframe."""
    if len(data) < 60:  # Need enough data for all indicators
        return data
        
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
    df['ichi_tk_cross'] = ichimoku_values['tenkan_cross_kijun'].astype(int)
    df['ichi_above_cloud'] = ichimoku_values['price_above_cloud'].astype(int)
    df['ichi_below_cloud'] = ichimoku_values['price_below_cloud'].astype(int)
    
    # Stochastic Oscillator
    stoch_values = calculate_stochastic(df)
    df['stoch_k'] = stoch_values['slowk']
    df['stoch_d'] = stoch_values['slowd']
    
    # ADX
    adx_values = calculate_adx(df)
    df['adx'] = adx_values['adx']
    df['plus_di'] = adx_values['plus_di']
    df['minus_di'] = adx_values['minus_di']
    
    # ATR
    df['atr'] = calculate_atr(df)
    
    # OBV
    if 'volume' in df.columns:
        df['obv'] = calculate_obv(df)
    
    # Pivot Points (standard method)
    pivot_values = calculate_pivot_points(df)
    df['pivot'] = pivot_values['pivot']
    df['pivot_r1'] = pivot_values['r1']
    df['pivot_s1'] = pivot_values['s1']
    
    # Fibonacci levels for larger timeframes
    if len(df) > 200:
        fib_values = calculate_fibonacci_retracement(df)
        df['fib_618'] = fib_values['uptrend_618']  # Golden ratio - key level
        df['fib_d_618'] = fib_values['downtrend_618']  # Golden ratio - key level
    
    # Support and Resistance
    sr_values = calculate_support_resistance(df)
    df['support'] = sr_values['support']
    df['resistance'] = sr_values['resistance']
    
    # Simple moving averages
    df['sma_10'] = talib.SMA(df['close'], timeperiod=10)
    df['sma_20'] = talib.SMA(df['close'], timeperiod=20)
    df['sma_50'] = talib.SMA(df['close'], timeperiod=50)
    df['sma_200'] = talib.SMA(df['close'], timeperiod=200)
    
    # Exponential moving averages
    df['ema_9'] = talib.EMA(df['close'], timeperiod=9)
    df['ema_21'] = talib.EMA(df['close'], timeperiod=21)
    
    # Trend detection
    df['uptrend'] = (df['sma_20'] > df['sma_50']).astype(int)
    df['downtrend'] = (df['sma_20'] < df['sma_50']).astype(int)
    
    return df
