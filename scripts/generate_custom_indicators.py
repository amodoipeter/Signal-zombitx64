#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to demonstrate generating and visualizing custom technical indicators.
"""

import os
import sys
import asyncio
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplfinance as mpf
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.market_data.fetcher import MarketDataFetcher
from app.core.ai.indicators import (
    calculate_all_indicators,
    calculate_ichimoku,
    calculate_fibonacci_retracement,
    calculate_pivot_points,
    calculate_support_resistance
)

async def fetch_and_analyze_symbol(symbol: str, timeframe: str) -> pd.DataFrame:
    """Fetch data for a symbol and calculate all indicators."""
    print(f"Fetching data for {symbol} on {timeframe} timeframe...")
    
    fetcher = MarketDataFetcher()
    data = await fetcher.fetch_data(symbol, timeframe, limit=200)
    await fetcher.close()
    
    if data.empty:
        print(f"No data available for {symbol}")
        return None
    
    print(f"Calculating indicators for {symbol}...")
    df_with_indicators = calculate_all_indicators(data)
    
    return df_with_indicators

def visualize_indicators(df: pd.DataFrame, symbol: str, timeframe: str):
    """Visualize technical indicators on a chart."""
    # Create a copy to avoid modifying the original
    df_plot = df.copy()
    
    # Prepare the plot
    fig = plt.figure(figsize=(15, 12))
    
    # Main price chart
    ax1 = plt.subplot2grid((6, 1), (0, 0), rowspan=2)
    ax1.set_title(f"{symbol} {timeframe} with Technical Indicators")
    ax1.plot(df_plot.index, df_plot['close'], label='Price')
    
    # Add Bollinger Bands
    ax1.plot(df_plot.index, df_plot['bb_upper'], 'r--', label='BB Upper')
    ax1.plot(df_plot.index, df_plot['bb_middle'], 'g-', label='BB Middle')
    ax1.plot(df_plot.index, df_plot['bb_lower'], 'r--', label='BB Lower')
    
    # Add Ichimoku Cloud
    ax1.plot(df_plot.index, df_plot['ichi_tenkan'], 'm-', label='Tenkan-sen')
    ax1.plot(df_plot.index, df_plot['ichi_kijun'], 'c-', label='Kijun-sen')
    
    # Add moving averages
    ax1.plot(df_plot.index, df_plot['sma_20'], 'y-', label='SMA 20')
    ax1.plot(df_plot.index, df_plot['sma_50'], 'orange', label='SMA 50')
    
    # Add Fibonacci for the last N periods
    last_100 = df_plot.iloc[-100:]
    high_point = last_100['high'].max()
    low_point = last_100['low'].min()
    fib_levels = [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1]
    colors = ['black', 'red', 'green', 'blue', 'purple', 'orange', 'gray']
    
    for level, color in zip(fib_levels, colors):
        value = low_point + level * (high_point - low_point)
        ax1.axhline(y=value, color=color, linestyle='--', alpha=0.5, 
                   label=f'Fib {level}')
    
    ax1.legend(loc='upper left')
    ax1.grid(True)
    
    # RSI chart
    ax2 = plt.subplot2grid((6, 1), (2, 0))
    ax2.set_title("RSI")
    ax2.plot(df_plot.index, df_plot['rsi'], 'b-')
    ax2.axhline(y=70, color='r', linestyle='--')
    ax2.axhline(y=30, color='g', linestyle='--')
    ax2.grid(True)
    ax2.set_ylim(0, 100)
    
    # MACD chart
    ax3 = plt.subplot2grid((6, 1), (3, 0))
    ax3.set_title("MACD")
    ax3.plot(df_plot.index, df_plot['macd'], 'b-', label='MACD')
    ax3.plot(df_plot.index, df_plot['macd_signal'], 'r-', label='Signal')
    ax3.bar(df_plot.index, df_plot['macd_hist'], color='g', alpha=0.5, width=0.01, label='Histogram')
    ax3.axhline(y=0, color='k', linestyle='-')
    ax3.legend(loc='upper left')
    ax3.grid(True)
    
    # Stochastic oscillator
    ax4 = plt.subplot2grid((6, 1), (4, 0))
    ax4.set_title("Stochastic")
    ax4.plot(df_plot.index, df_plot['stoch_k'], 'b-', label='%K')
    ax4.plot(df_plot.index, df_plot['stoch_d'], 'r-', label='%D')
    ax4.axhline(y=80, color='r', linestyle='--')
    ax4.axhline(y=20, color='g', linestyle='--')
    ax4.legend(loc='upper left')
    ax4.grid(True)
    ax4.set_ylim(0, 100)
    
    # ADX
    ax5 = plt.subplot2grid((6, 1), (5, 0))
    ax5.set_title("ADX")
    ax5.plot(df_plot.index, df_plot['adx'], 'b-', label='ADX')
    ax5.plot(df_plot.index, df_plot['plus_di'], 'g-', label='+DI')
    ax5.plot(df_plot.index, df_plot['minus_di'], 'r-', label='-DI')
    ax5.axhline(y=25, color='k', linestyle='--')
    ax5.legend(loc='upper left')
    ax5.grid(True)
    ax5.set_ylim(0, 100)
    
    plt.tight_layout()
    
    # Save the chart
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"{symbol}_{timeframe}_indicators.png"))
    print(f"Chart saved to output/{symbol}_{timeframe}_indicators.png")
    
    plt.close()

    # Create a candlestick chart with Ichimoku Cloud using mplfinance
    ic = mpf.make_addplot(df_plot[['ichi_tenkan', 'ichi_kijun']])
    
    # Set up the Ichimoku Cloud fill
    cloud_green = df_plot['ichi_senkou_a'] > df_plot['ichi_senkou_b']
    cloud_red = df_plot['ichi_senkou_a'] <= df_plot['ichi_senkou_b']
    
    ap = [
        ic,
        mpf.make_addplot(df_plot['ichi_senkou_a'], color='g', alpha=0.3),
        mpf.make_addplot(df_plot['ichi_senkou_b'], color='r', alpha=0.3)
    ]
    
    # Create mplfinance figure
    fig, axes = mpf.plot(
        df_plot.iloc[-60:],  # Last 60 candles
        type='candle',
        style='charles',
        title=f"{symbol} {timeframe} with Ichimoku Cloud",
        figsize=(15, 10),
        addplot=ap,
        returnfig=True
    )
    
    # Save the candlestick chart
    plt.savefig(os.path.join(output_dir, f"{symbol}_{timeframe}_candlestick.png"))
    print(f"Candlestick chart saved to output/{symbol}_{timeframe}_candlestick.png")
    
    plt.close()

async def main():
    """Main function to demonstrate indicator generation and visualization."""
    # List of symbols and timeframes to analyze
    symbols = ["BTCUSDT", "ETHUSDT", "EURUSD"]
    timeframes = ["1h", "4h"]
    
    for symbol in symbols:
        for timeframe in timeframes:
            df = await fetch_and_analyze_symbol(symbol, timeframe)
            if df is not None:
                visualize_indicators(df, symbol, timeframe)
                
                # Print some key indicator values for the most recent candle
                print(f"\nRecent indicator values for {symbol} {timeframe}:")
                print(f"RSI: {df['rsi'].iloc[-1]:.2f}")
                print(f"MACD: {df['macd'].iloc[-1]:.6f}")
                print(f"MACD Signal: {df['macd_signal'].iloc[-1]:.6f}")
                print(f"Bollinger Upper: {df['bb_upper'].iloc[-1]:.6f}")
                print(f"Bollinger Middle: {df['bb_middle'].iloc[-1]:.6f}")
                print(f"Bollinger Lower: {df['bb_lower'].iloc[-1]:.6f}")
                print(f"Ichimoku Tenkan-sen: {df['ichi_tenkan'].iloc[-1]:.6f}")
                print(f"Ichimoku Kijun-sen: {df['ichi_kijun'].iloc[-1]:.6f}")
                print(f"Ichimoku Senkou Span A: {df['ichi_senkou_a'].iloc[-1]:.6f}")
                print(f"Ichimoku Senkou Span B: {df['ichi_senkou_b'].iloc[-1]:.6f}")
                print(f"ADX: {df['adx'].iloc[-1]:.2f}")
                print(f"Stochastic %K: {df['stoch_k'].iloc[-1]:.2f}")
                print(f"Stochastic %D: {df['stoch_d'].iloc[-1]:.2f}")
                print("\n" + "="*50 + "\n")

if __name__ == "__main__":
    asyncio.run(main())
