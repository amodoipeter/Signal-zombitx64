import io
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import mplfinance as mpf
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional
import logging

from app.core.market_data.fetcher import MarketDataFetcher
from app.core.ai.indicators import calculate_all_indicators

logger = logging.getLogger(__name__)

async def generate_chart_image(symbol: str, timeframe: str) -> Optional[io.BytesIO]:
    """
    Generate a chart image for a symbol and timeframe.
    Returns a BytesIO object that can be sent to Telegram.
    """
    try:
        # Fetch market data
        fetcher = MarketDataFetcher()
        df = await fetcher.fetch_data(symbol, timeframe, limit=100)
        await fetcher.close()
        
        if df.empty:
            logger.warning(f"No data available for chart generation: {symbol} {timeframe}")
            return None
        
        # Calculate indicators
        df_with_indicators = calculate_all_indicators(df)
        
        # Create figure
        fig = plt.figure(figsize=(12, 8))
        
        # Plot candlestick chart
        ax1 = plt.subplot2grid((4, 1), (0, 0), rowspan=2, fig=fig)
        mpf.plot(df, type='candle', style='charles', ax=ax1, volume=False)
        
        # Add Bollinger Bands to main chart
        ax1.plot(df.index, df_with_indicators['bb_upper'], 'b--', alpha=0.5)
        ax1.plot(df.index, df_with_indicators['bb_middle'], 'g-', alpha=0.5)
        ax1.plot(df.index, df_with_indicators['bb_lower'], 'b--', alpha=0.5)
        
        # Add Ichimoku Cloud to main chart
        ax1.plot(df.index, df_with_indicators['ichi_tenkan'], 'r-', alpha=0.7, label='Tenkan')
        ax1.plot(df.index, df_with_indicators['ichi_kijun'], 'b-', alpha=0.7, label='Kijun')
        
        # Fill Ichimoku Cloud
        ax1.fill_between(df.index,
                         df_with_indicators['ichi_senkou_a'],
                         df_with_indicators['ichi_senkou_b'],
                         where=df_with_indicators['ichi_senkou_a'] >= df_with_indicators['ichi_senkou_b'],
                         color='green', alpha=0.2)
        ax1.fill_between(df.index,
                         df_with_indicators['ichi_senkou_a'],
                         df_with_indicators['ichi_senkou_b'],
                         where=df_with_indicators['ichi_senkou_a'] < df_with_indicators['ichi_senkou_b'],
                         color='red', alpha=0.2)
        
        ax1.legend(loc='upper left')
        ax1.set_title(f'{symbol} {timeframe} Chart')
        
        # Plot RSI
        ax2 = plt.subplot2grid((4, 1), (2, 0), rowspan=1, fig=fig, sharex=ax1)
        ax2.plot(df.index, df_with_indicators['rsi'], 'b-')
        ax2.axhline(y=30, color='g', linestyle='--')
        ax2.axhline(y=70, color='r', linestyle='--')
        ax2.set_ylabel('RSI')
        
        # Plot MACD
        ax3 = plt.subplot2grid((4, 1), (3, 0), rowspan=1, fig=fig, sharex=ax1)
        ax3.plot(df.index, df_with_indicators['macd'], 'b-', label='MACD')
        ax3.plot(df.index, df_with_indicators['macd_signal'], 'r-', label='Signal')
        ax3.bar(df.index, df_with_indicators['macd_hist'], color='g', alpha=0.5, width=0.01)
        ax3.legend(loc='upper left')
        ax3.set_ylabel('MACD')
        
        # Format date axis
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax3.xaxis.set_major_locator(mdates.DayLocator(interval=5))
        plt.xticks(rotation=45)
        
        # Adjust layout and save to buffer
        plt.tight_layout()
        
        # Save figure to BytesIO object
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        
        return buf
        
    except Exception as e:
        logger.error(f"Error generating chart image: {str(e)}")
        return None
