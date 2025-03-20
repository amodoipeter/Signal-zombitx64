import aiohttp
import pandas as pd
import numpy as np
import logging
import ccxt.async_support as ccxt
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any

from app.core.config import settings

logger = logging.getLogger(__name__)

class MarketDataFetcher:
    def __init__(self):
        """Initialize the market data fetcher with exchange connections."""
        self.binance = ccxt.binance({
            'apiKey': settings.BINANCE_API_KEY,
            'secret': settings.BINANCE_API_SECRET,
            'enableRateLimit': True,
        })
        
        # Timeframe mapping to milliseconds
        self.timeframe_ms = {
            '1m': 60 * 1000,
            '5m': 5 * 60 * 1000,
            '15m': 15 * 60 * 1000,
            '30m': 30 * 60 * 1000,
            '1h': 60 * 60 * 1000,
            '4h': 4 * 60 * 60 * 1000,
            '1d': 24 * 60 * 60 * 1000,
        }
    
    async def fetch_data(self, symbol: str, timeframe: str, limit: int = 200) -> pd.DataFrame:
        """Fetch market data for a symbol and timeframe."""
        try:
            # Fetch OHLCV data
            ohlcv = await self.binance.fetch_ohlcv(symbol, timeframe, limit=limit)
            
            if not ohlcv or len(ohlcv) == 0:
                logger.warning(f"No data returned for {symbol} on {timeframe}")
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Convert values to float
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching market data for {symbol} {timeframe}: {str(e)}")
            return pd.DataFrame()
    
    async def close(self):
        """Close exchange connections."""
        await self.binance.close()
