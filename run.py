#!/usr/bin/env python
"""
Main script to run the AI Signal Provider application.
This script initializes and starts all necessary components.
"""

import os
import sys
import logging
import argparse
import threading
import time
from dotenv import load_dotenv

# Add the app directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

# Import project modules
from app.main import app as fastapi_app
from app.bot_integration.telegram_bot import TelegramBotManager
from app.data_collection.market_data import MarketDataCollector
from app.signal_generation.signal_generator import SignalGenerator
from app.monitoring.performance_tracker import PerformanceTracker
from app.database.db_manager import DBManager
import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def run_fastapi():
    """Run the FastAPI application."""
    logger.info("Starting FastAPI server...")
    uvicorn.run(
        "app.main:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", 8000)),
        reload=os.getenv("API_RELOAD", "false").lower() == "true"
    )

def run_telegram_bot(db_manager):
    """Run the Telegram bot."""
    logger.info("Starting Telegram bot...")
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not bot_token:
        logger.error("TELEGRAM_BOT_TOKEN not found in environment variables")
        return
    
    bot_manager = TelegramBotManager(bot_token, db_manager)
    bot_manager.start()

def run_data_collection(db_manager):
    """Run the data collection process."""
    logger.info("Starting data collection...")
    collector = MarketDataCollector()
    
    # Get symbols to monitor from environment variables or use defaults
    symbols_str = os.getenv("MONITORED_SYMBOLS", "BTC/USDT,ETH/USDT")
    symbols = [symbol.strip() for symbol in symbols_str.split(",")]
    
    # Collect data in a loop
    interval = int(os.getenv("DATA_COLLECTION_INTERVAL_SECONDS", 3600))  # Default to hourly
    
    while True:
        try:
            for symbol in symbols:
                logger.info(f"Collecting data for {symbol}")
                historical_data = collector.fetch_historical_data(symbol, "1h", limit=1000)
                
                # Convert data to the format expected by DBManager
                market_data = []
                for candle in historical_data:
                    market_data.append({
                        "symbol": symbol,
                        "timestamp": candle[0],  # timestamp
                        "open": candle[1],
                        "high": candle[2],
                        "low": candle[3],
                        "close": candle[4],
                        "volume": candle[5]
                    })
                
                # Save to database
                if market_data:
                    db_manager.save_market_data(market_data)
                    logger.info(f"Saved {len(market_data)} candles for {symbol}")
            
            # Sleep until next collection
            logger.info(f"Data collection complete. Sleeping for {interval} seconds")
            time.sleep(interval)
        except Exception as e:
            logger.error(f"Error in data collection: {str(e)}")
            time.sleep(60)  # Sleep for a minute before retrying

def run_signal_generator(db_manager):
    """Run the signal generation process."""
    logger.info("Starting signal generation...")
    signal_generator = SignalGenerator()
    performance_tracker = PerformanceTracker(db_manager)
    
    # Get symbols to monitor from environment variables or use defaults
    symbols_str = os.getenv("MONITORED_SYMBOLS", "BTC/USDT,ETH/USDT")
    symbols = [symbol.strip() for symbol in symbols_str.split(",")]
    
    # Generate signals in a loop
    interval = int(os.getenv("SIGNAL_GENERATION_INTERVAL_SECONDS", 3600))  # Default to hourly
    
    while True:
        try:
            for symbol in symbols:
                logger.info(f"Generating signals for {symbol}")
                
                # Get historical data from database
                now = time.time()
                start_timestamp = now - (1000 * 3600)  # Get last 1000 hours
                
                # In a real implementation, you would get the data from the database
                # Here we'll use the MarketDataCollector directly for simplicity
                collector = MarketDataCollector()
                data = collector.fetch_historical_data(symbol, "1h", limit=1000)
                
                # Process data if available
                if data:
                    # Train the model if needed
                    signal_generator.train_model(data)
                    
                    # Generate signals
                    signals = signal_generator.generate_signals(data)
                    
                    if signals:
                        latest_signal = signals[-1]
                        
                        # Save the signal to the database
                        signal_data = {
                            "symbol": symbol,
                            "signal_type": latest_signal["signal"],
                            "confidence": latest_signal["confidence"],
                            "price": latest_signal["price"],
                            "timestamp": latest_signal["timestamp"],
                            "is_sent": False
                        }
                        
                        # Create signal in database
                        db_manager.create_signal(signal_data)
                        
                        # Record the signal in the performance tracker
                        performance_tracker.record_signal(signal_data)
                        
                        logger.info(f"Generated signal for {symbol}: {latest_signal['signal']} at {latest_signal['price']}")
            
            # Sleep until next generation
            logger.info(f"Signal generation complete. Sleeping for {interval} seconds")
            time.sleep(interval)
        except Exception as e:
            logger.error(f"Error in signal generation: {str(e)}")
            time.sleep(60)  # Sleep for a minute before retrying

def run_dashboard():
    """Run the Streamlit dashboard."""
    logger.info("Starting Streamlit dashboard...")
    dashboard_port = os.getenv("DASHBOARD_PORT", "8501")
    
    # Run the Streamlit app using streamlit run
    os.system(f"streamlit run app/dashboard/dashboard.py --server.port={dashboard_port}")


def main():
    """Main function to start all components."""
    parser = argparse.ArgumentParser(description="Run AI Signal Provider components")
    parser.add_argument("--api", action="store_true", help="Run the FastAPI server")
    parser.add_argument("--bot", action="store_true", help="Run the Telegram bot")
    parser.add_argument("--data", action="store_true", help="Run the data collection")
    parser.add_argument("--signals", action="store_true", help="Run the signal generation")
    parser.add_argument("--dashboard", action="store_true", help="Run the Streamlit dashboard")
    parser.add_argument("--all", action="store_true", help="Run all components")
    
    args = parser.parse_args()
    
    # If no arguments are provided, show help
    if not any(vars(args).values()):
        parser.print_help()
        return
    
    # Initialize the database manager
    try:
        db_manager = DBManager()
        db_manager.create_tables()
        logger.info("Database initialized")
    except Exception as e:
        logger.error(f"Failed to initialize database: {str(e)}")
        return
    
    # List to keep track of threads
    threads = []
    
    # Start FastAPI server
    if args.api or args.all:
        api_thread = threading.Thread(target=run_fastapi)
        api_thread.daemon = True
        api_thread.start()
        threads.append(api_thread)
    
    # Start Telegram bot
    if args.bot or args.all:
        bot_thread = threading.Thread(target=run_telegram_bot, args=(db_manager,))
        bot_thread.daemon = True
        bot_thread.start()
        threads.append(bot_thread)
    
    # Start data collection
    if args.data or args.all:
        data_thread = threading.Thread(target=run_data_collection, args=(db_manager,))
        data_thread.daemon = True
        data_thread.start()
        threads.append(data_thread)
    
    # Start signal generation
    if args.signals or args.all:
        signals_thread = threading.Thread(target=run_signal_generator, args=(db_manager,))
        signals_thread.daemon = True
        signals_thread.start()
        threads.append(signals_thread)
    
    # Start Streamlit dashboard
    if args.dashboard or args.all:
        dashboard_thread = threading.Thread(target=run_dashboard)
        dashboard_thread.daemon = True
        dashboard_thread.start()
        threads.append(dashboard_thread)
    
    # Wait for all threads to complete
    try:
        while True:
            # Check if any thread has died
            for thread in threads:
                if not thread.is_alive():
                    logger.error(f"Thread {thread.name} has died")
            
            time.sleep(60)
    except KeyboardInterrupt:
        logger.info("Received KeyboardInterrupt. Shutting down...")


if __name__ == "__main__":
    main()
