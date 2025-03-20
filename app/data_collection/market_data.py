"""
Market data collection module for the AI Signal Provider.
This module handles fetching and storing market data from various exchanges.
"""

import os
import logging
import pandas as pd
import ccxt
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Configure logging
logger = logging.getLogger(__name__)


class MarketDataCollector:
    """
    A class to collect and process market data from various exchanges.
    Supports both historical data retrieval and real-time data streaming.
    """

    def __init__(self, exchange_id="binance", timeframe="1h"):
        """
        Initialize the MarketDataCollector.
        
        Args:
            exchange_id (str): The exchange to connect to (default: 'binance')
            timeframe (str): The timeframe for candlestick data (default: '1h')
        """
        load_dotenv()
        
        self.exchange_id = exchange_id
        self.timeframe = timeframe
        self.exchange = self._initialize_exchange()
        
    def _initialize_exchange(self):
        """
        Initialize the exchange connection.
        
        Returns:
            ccxt.Exchange: An instance of the exchange
        """
        exchange_class = getattr(ccxt, self.exchange_id)
        
        # Get API credentials from environment variables
        api_key = os.getenv(f"{self.exchange_id.upper()}_API_KEY")
        api_secret = os.getenv(f"{self.exchange_id.upper()}_API_SECRET")
        
        exchange_params = {}
        if api_key and api_secret:
            exchange_params = {
                'apiKey': api_key,
                'secret': api_secret,
                'enableRateLimit': True
            }
        
        exchange = exchange_class(exchange_params)
        logger.info(f"Initialized {self.exchange_id} exchange connection")
        return exchange
    
    def fetch_historical_data(self, symbol, days_back=30):
        """
        Fetch historical candlestick data for a specific symbol.
        
        Args:
            symbol (str): The trading pair symbol (e.g., 'BTC/USDT')
            days_back (int): Number of days to look back (default: 30)
            
        Returns:
            pd.DataFrame: DataFrame containing the historical market data
        """
        try:
            since = int((datetime.now() - timedelta(days=days_back)).timestamp() * 1000)
            
            logger.info(f"Fetching {days_back} days of historical data for {symbol}")
            candles = self.exchange.fetch_ohlcv(symbol, timeframe=self.timeframe, since=since)
            
            # Convert to DataFrame
            df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            logger.info(f"Successfully fetched {len(df)} candles for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {str(e)}")
            raise
    
    def fetch_real_time_data(self, symbol):
        """
        Fetch the latest market data for a specific symbol.
        
        Args:
            symbol (str): The trading pair symbol (e.g., 'BTC/USDT')
            
        Returns:
            dict: The latest ticker data
        """
        try:
            logger.info(f"Fetching real-time data for {symbol}")
            ticker = self.exchange.fetch_ticker(symbol)
            logger.info(f"Successfully fetched real-time data for {symbol}")
            return ticker
        except Exception as e:
            logger.error(f"Error fetching real-time data for {symbol}: {str(e)}")
            raise
    
    def get_supported_symbols(self):
        """
        Get a list of all supported trading symbols from the exchange.
        
        Returns:
            list: List of supported symbols
        """
        try:
            markets = self.exchange.load_markets()
            symbols = list(markets.keys())
            return symbols
        except Exception as e:
            logger.error(f"Error fetching supported symbols: {str(e)}")
            raise


# Example usage
if __name__ == "__main__":
    collector = MarketDataCollector()
    symbols = collector.get_supported_symbols()
    print(f"Supported symbols: {len(symbols)}")
    
    # Fetch historical data for BTC/USDT
    btc_data = collector.fetch_historical_data("BTC/USDT", days_back=7)
    print(btc_data.head())
    
    # Fetch real-time data for BTC/USDT
    btc_ticker = collector.fetch_real_time_data("BTC/USDT")
    print(f"Current BTC price: {btc_ticker['last']}")
