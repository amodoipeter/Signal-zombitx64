"""
Telegram Bot Integration for the AI Signal Provider.
This module handles the sending of trading signals to subscribers via Telegram.
"""

import os
import logging
import asyncio
from datetime import datetime
import io
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from telegram import Update, ParseMode, InputFile
from telegram.ext import (
    Updater,
    CommandHandler,
    CallbackContext,
    MessageHandler,
    Filters,
    ConversationHandler
)
from dotenv import load_dotenv

# Configure logging
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class TelegramBotManager:
    """
    A class to manage a Telegram bot for sending trading signals to subscribers.
    """
    
    def __init__(self):
        """
        Initialize the TelegramBotManager.
        """
        self.token = os.getenv("TELEGRAM_BOT_TOKEN")
        if not self.token:
            raise ValueError("Telegram bot token not found in environment variables")
        
        self.updater = Updater(token=self.token, use_context=True)
        self.dispatcher = self.updater.dispatcher
        
        # For tracking active signals
        self.active_signals = {}  # {signal_id: signal_data}
        
        # Register handlers
        self._register_handlers()
        
        logger.info("Telegram bot initialized")
    
    def _register_handlers(self):
        """
        Register command handlers for the Telegram bot.
        """
        # Command handlers
        self.dispatcher.add_handler(CommandHandler("start", self._start_command))
        self.dispatcher.add_handler(CommandHandler("help", self._help_command))
        self.dispatcher.add_handler(CommandHandler("subscribe", self._subscribe_command))
        self.dispatcher.add_handler(CommandHandler("unsubscribe", self._unsubscribe_command))
        self.dispatcher.add_handler(CommandHandler("status", self._status_command))
        
        # Error handler
        self.dispatcher.add_error_handler(self._error_handler)
    
    def start(self):
        """
        Start the Telegram bot.
        """
        self.updater.start_polling()
        logger.info("Telegram bot started")
    
    def stop(self):
        """
        Stop the Telegram bot.
        """
        self.updater.stop()
        logger.info("Telegram bot stopped")
    
    async def send_signal(self, chat_id, signal_data):
        """
        Send a trading signal to a specific chat.
        
        Args:
            chat_id (int/str): Telegram chat ID to send the signal to
            signal_data (dict): Dictionary containing signal information
        """
        try:
            # Format the signal message
            message = self._format_signal_message(signal_data)
            
            # Send the message
            await self.updater.bot.send_message(
                chat_id=chat_id,
                text=message,
                parse_mode=ParseMode.MARKDOWN
            )
            
            logger.info(f"Signal sent to chat {chat_id}")
            return True
        except Exception as e:
            logger.error(f"Error sending signal to chat {chat_id}: {str(e)}")
            return False
    
    async def broadcast_signal(self, subscriber_ids, signal_data):
        """
        Broadcast a trading signal to multiple subscribers.
        
        Args:
            subscriber_ids (list): List of Telegram chat IDs to send the signal to
            signal_data (dict): Dictionary containing signal information
            
        Returns:
            dict: A dictionary with success and failure counts
        """
        results = {"success": 0, "failed": 0}
        
        for chat_id in subscriber_ids:
            success = await self.send_signal(chat_id, signal_data)
            if success:
                results["success"] += 1
            else:
                results["failed"] += 1
        
        logger.info(f"Signal broadcast complete: {results}")
        return results
    
    def _format_signal_message(self, signal_data):
        """
        Format a trading signal into a readable message.
        
        Args:
            signal_data (dict): Dictionary containing signal information
            
        Returns:
            str: Formatted message string
        """
        # Extract signal data with defaults
        symbol = signal_data.get("symbol", "Unknown")
        signal = signal_data.get("signal", "NEUTRAL")
        confidence = signal_data.get("confidence", 0.0)
        price = signal_data.get("price", 0.0)
        timestamp = signal_data.get("timestamp", datetime.now().isoformat())
        tp_level = signal_data.get("take_profit")
        sl_level = signal_data.get("stop_loss")
        
        # Format timestamp
        if isinstance(timestamp, str):
            try:
                timestamp = datetime.fromisoformat(timestamp)
            except ValueError:
                timestamp = datetime.now()
        
        # Format the message
        emoji = "🔴" if signal == "SELL" else "🟢" if signal == "BUY" else "⚪"
        
        message = (
            f"*{emoji} {signal} SIGNAL: {symbol} {emoji}*\n\n"
            f"*Price:* `{price}`\n"
            f"*Confidence:* `{confidence:.2%}`\n"
            f"*Time:* `{timestamp.strftime('%Y-%m-%d %H:%M:%S')}`\n\n"
        )
        
        # Add take profit and stop loss if available
        if tp_level and sl_level:
            if signal == "BUY":
                risk_reward = (tp_level - price) / (price - sl_level)
                message += (
                    f"*Take Profit:* `{tp_level:.2f}`\n"
                    f"*Stop Loss:* `{sl_level:.2f}`\n"
                    f"*Risk/Reward:* `{risk_reward:.2f}`\n\n"
                )
            elif signal == "SELL":
                risk_reward = (price - tp_level) / (sl_level - price)
                message += (
                    f"*Take Profit:* `{tp_level:.2f}`\n"
                    f"*Stop Loss:* `{sl_level:.2f}`\n"
                    f"*Risk/Reward:* `{risk_reward:.2f}`\n\n"
                )
        
        # Add AI analysis if available
        if signal_data.get("technical_indicators"):
            ti = signal_data["technical_indicators"]
            message += "*Technical Analysis:*\n"
            
            # RSI
            rsi = ti.get("rsi")
            if rsi is not None:
                rsi_status = "Oversold" if rsi < 30 else "Overbought" if rsi > 70 else "Neutral"
                message += f"RSI: `{rsi:.2f}` ({rsi_status})\n"
            
            # MACD
            macd = ti.get("macd")
            macd_signal = ti.get("macd_signal")
            if macd is not None and macd_signal is not None:
                macd_status = "Bullish" if macd > macd_signal else "Bearish"
                message += f"MACD: `{macd:.4f}` ({macd_status})\n"
            
            # Bollinger Bands
            if "bollinger_upper" in ti and "bollinger_lower" in ti:
                band_width = ti["bollinger_upper"] - ti["bollinger_lower"]
                volatility = "High" if band_width > (price * 0.03) else "Low"
                message += f"Volatility: {volatility}\n\n"
        
        # Add recommendation based on signal type
        if signal == "BUY":
            message += (
                f"*Recommendation:*\n"
                f"✅ Consider entering a long position\n"
                f"✅ Set stop loss at indicated level\n"
                f"✅ Target profit at indicated level\n"
            )
        elif signal == "SELL":
            message += (
                f"*Recommendation:*\n"
                f"✅ Consider entering a short position\n"
                f"✅ Set stop loss at indicated level\n"
                f"✅ Target profit at indicated level\n"
            )
        else:
            message += "*Recommendation:* ⚠️ Wait for a clearer signal\n"
        
        message += "\n_This is an automated signal and not financial advice. Always do your own research before trading._"
        
        return message
    
    async def generate_signal_chart(self, signal_data, historical_data):
        """
        Generate a chart visualization for the signal.
        
        Args:
            signal_data (dict): Dictionary containing signal information
            historical_data (pd.DataFrame): DataFrame with historical price data
            
        Returns:
            io.BytesIO: Image buffer containing the chart
        """
        try:
            # Create figure and axis
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Plot price data
            ax.plot(historical_data.index, historical_data['close'], label='Price')
            
            # Mark the signal
            current_price = signal_data['price']
            signal_date = historical_data.index[-1]
            
            # Add markers and annotations for Buy/Sell signals
            if signal_data['signal'] == 'BUY':
                ax.plot(signal_date, current_price, 'go', markersize=10)
                ax.annotate('BUY', (signal_date, current_price), 
                            xytext=(10, 15), textcoords='offset points',
                            color='green', fontweight='bold')
                
                # Add take profit and stop loss levels if available
                if signal_data.get('take_profit') and signal_data.get('stop_loss'):
                    tp = signal_data['take_profit']
                    sl = signal_data['stop_loss']
                    ax.axhline(y=tp, color='g', linestyle='--', alpha=0.6, label=f'TP: {tp:.2f}')
                    ax.axhline(y=sl, color='r', linestyle='--', alpha=0.6, label=f'SL: {sl:.2f}')
                    
            elif signal_data['signal'] == 'SELL':
                ax.plot(signal_date, current_price, 'ro', markersize=10)
                ax.annotate('SELL', (signal_date, current_price), 
                            xytext=(10, 15), textcoords='offset points',
                            color='red', fontweight='bold')
                
                # Add take profit and stop loss levels if available
                if signal_data.get('take_profit') and signal_data.get('stop_loss'):
                    tp = signal_data['take_profit']
                    sl = signal_data['stop_loss']
                    ax.axhline(y=tp, color='g', linestyle='--', alpha=0.6, label=f'TP: {tp:.2f}')
                    ax.axhline(y=sl, color='r', linestyle='--', alpha=0.6, label=f'SL: {sl:.2f}')
            
            # Add technical indicators
            if 'technical_indicators' in signal_data:
                ti = signal_data['technical_indicators']
                if 'bollinger_upper' in ti and 'bollinger_lower' in ti and 'bollinger_mid' in ti:
                    # Get last few bollinger band values for plotting
                    if 'bollinger_upper' in historical_data.columns:
                        ax.plot(historical_data.index, historical_data['bollinger_upper'], 'b--', alpha=0.5, label='Upper BB')
                        ax.plot(historical_data.index, historical_data['bollinger_mid'], 'b-', alpha=0.5, label='Middle BB')
                        ax.plot(historical_data.index, historical_data['bollinger_lower'], 'b--', alpha=0.5, label='Lower BB')
            
            # Set title and labels
            symbol = signal_data['symbol']
            confidence = signal_data['confidence']
            ax.set_title(f"{symbol} Signal ({signal_data['signal']}) - Confidence: {confidence:.2%}")
            ax.set_xlabel('Date')
            ax.set_ylabel('Price')
            
            # Add legend
            ax.legend()
            
            # Set grid
            ax.grid(alpha=0.3)
            
            # Ensure dates are formatted properly
            fig.autofmt_xdate()
            
            # Save to buffer
            buf = io.BytesIO()
            plt.tight_layout()
            fig.savefig(buf, format='png', dpi=150)
            buf.seek(0)
            
            # Close the figure to free memory
            plt.close(fig)
            
            return buf
        except Exception as e:
            logger.error(f"Error generating signal chart: {str(e)}")
            return None
    
    async def send_signal_with_chart(self, chat_id, signal_data, historical_data=None):
        """
        Send a trading signal with chart visualization to a specific chat.
        
        Args:
            chat_id (int/str): Telegram chat ID 
            signal_data (dict): Dictionary containing signal information
            historical_data (pd.DataFrame, optional): Historical price data for chart generation
            
        Returns:
            bool: Success status
        """
        try:
            # Generate signal message
            message = self._format_signal_message(signal_data)
            
            # Store signal for tracking
            signal_id = f"{signal_data['symbol']}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            self.active_signals[signal_id] = {
                **signal_data,
                "signal_id": signal_id,
                "chat_id": chat_id,
                "message_sent": True,
                "tp_hit": False,
                "sl_hit": False
            }
            
            # If we have historical data, generate and send chart
            if historical_data is not None and not historical_data.empty:
                chart_buffer = await self.generate_signal_chart(signal_data, historical_data)
                
                if chart_buffer:
                    # Send photo with caption
                    await self.updater.bot.send_photo(
                        chat_id=chat_id, 
                        photo=InputFile(chart_buffer, filename=f"{signal_data['symbol']}_signal.png"),
                        caption=message,
                        parse_mode=ParseMode.MARKDOWN
                    )
                else:
                    # Fallback to text-only if chart generation failed
                    await self.updater.bot.send_message(
                        chat_id=chat_id,
                        text=message,
                        parse_mode=ParseMode.MARKDOWN
                    )
            else:
                # Send text-only message
                await self.updater.bot.send_message(
                    chat_id=chat_id,
                    text=message,
                    parse_mode=ParseMode.MARKDOWN
                )
            
            logger.info(f"Signal with ID {signal_id} sent to chat {chat_id}")
            return True
        except Exception as e:
            logger.error(f"Error sending signal to chat {chat_id}: {str(e)}")
            return False
    
    async def notify_tp_sl_hit(self, signal_id, hit_type, current_price):
        """
        Send a notification when a take profit or stop loss level is hit.
        
        Args:
            signal_id (str): ID of the signal
            hit_type (str): Either "TP" or "SL"
            current_price (float): Current price at which TP/SL was hit
            
        Returns:
            bool: Success status
        """
        if signal_id not in self.active_signals:
            logger.warning(f"Cannot notify for unknown signal ID: {signal_id}")
            return False
            
        signal = self.active_signals[signal_id]
        chat_id = signal["chat_id"]
        
        # Mark TP/SL as hit
        if hit_type == "TP":
            signal["tp_hit"] = True
            emoji = "🎯"
            color = "green"
            profit_loss = "PROFIT"
        else:  # SL
            signal["sl_hit"] = True
            emoji = "🛑"
            color = "red"
            profit_loss = "LOSS"
            
        symbol = signal["symbol"]
        original_price = signal["price"]
        price_change = (current_price - original_price) / original_price * 100
        
        # Create notification message
        message = (
            f"*{emoji} {hit_type} HIT: {symbol} {emoji}*\n\n"
            f"*Signal Type:* {signal['signal']}\n"
            f"*Entry Price:* `{original_price:.5f}`\n"
            f"*Current Price:* `{current_price:.5f}`\n"
            f"*Price Change:* `{price_change:.2f}%`\n"
            f"*{profit_loss} TAKEN*\n\n"
            f"_This signal is now closed._"
        )
        
        try:
            await self.updater.bot.send_message(
                chat_id=chat_id,
                text=message,
                parse_mode=ParseMode.MARKDOWN
            )
            logger.info(f"{hit_type} hit notification sent for signal {signal_id}")
            return True
        except Exception as e:
            logger.error(f"Error sending {hit_type} hit notification: {str(e)}")
            return False
    
    # Command handlers
    async def _start_command(self, update: Update, context: CallbackContext):
        """
        Handle the /start command.
        """
        user = update.effective_user
        message = (
            f"Hello {user.first_name}! 👋\n\n"
            f"Welcome to the AI Trading Signal Provider Bot. "
            f"This bot provides AI-generated trading signals for cryptocurrency, forex, and stock markets.\n\n"
            f"Use /help to see available commands."
        )
        await update.message.reply_text(message)
    
    async def _help_command(self, update: Update, context: CallbackContext):
        """
        Handle the /help command.
        """
        message = (
            "*Available Commands:*\n\n"
            "/start - Start the bot\n"
            "/help - Show this help message\n"
            "/subscribe - Subscribe to trading signals\n"
            "/unsubscribe - Unsubscribe from trading signals\n"
            "/status - Check your subscription status\n"
        )
        await update.message.reply_text(message, parse_mode=ParseMode.MARKDOWN)
    
    async def _subscribe_command(self, update: Update, context: CallbackContext):
        """
        Handle the /subscribe command.
        """
        user = update.effective_user
        chat_id = update.effective_chat.id
        
        # Here, you would add the user to your subscription database
        # For now, we'll just acknowledge the request
        
        message = (
            f"Thanks for subscribing, {user.first_name}! 🎉\n\n"
            f"You will now receive trading signals in this chat.\n"
            f"Use /unsubscribe to stop receiving signals at any time."
        )
        await update.message.reply_text(message)
    
    async def _unsubscribe_command(self, update: Update, context: CallbackContext):
        """
        Handle the /unsubscribe command.
        """
        user = update.effective_user
        chat_id = update.effective_chat.id
        
        # Here, you would remove the user from your subscription database
        # For now, we'll just acknowledge the request
        
        message = (
            f"You have been unsubscribed, {user.first_name}. 😢\n\n"
            f"You will no longer receive trading signals.\n"
            f"Use /subscribe to start receiving signals again."
        )
        await update.message.reply_text(message)
    
    async def _status_command(self, update: Update, context: CallbackContext):
        """
        Handle the /status command.
        """
        user = update.effective_user
        chat_id = update.effective_chat.id
        
        # Here, you would check the user's subscription status in your database
        # For now, we'll just provide a mock response
        
        message = (
            f"*Subscription Status*\n\n"
            f"User: {user.first_name} {user.last_name or ''}\n"
            f"Status: Active\n"
            f"Plan: Premium\n"
            f"Expiry: 2023-12-31\n"
        )
        await update.message.reply_text(message, parse_mode=ParseMode.MARKDOWN)
    
    async def _error_handler(self, update: Update, context: CallbackContext):
        """
        Handle errors in the Telegram bot.
        """
        logger.error(f"Update {update} caused error {context.error}")
        
        # If we have an update object, we can try to notify the user
        if update and update.effective_message:
            await update.effective_message.reply_text(
                "Sorry, an error occurred while processing your request. Please try again later."
            )


# Example usage
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize and start the bot
    bot_manager = TelegramBotManager()
    bot_manager.start()
    
    # Example signal
    example_signal = {
        "symbol": "BTC/USDT",
        "signal": "BUY",
        "confidence": 0.85,
        "price": 30000.00,
        "timestamp": datetime.now().isoformat(),
        "take_profit": 32000.00,
        "stop_loss": 28000.00,
        "technical_indicators": {
            "rsi": 50.0,
            "macd": 0.001,
            "macd_signal": 0.0005,
            "bollinger_upper": 31000.00,
            "bollinger_mid": 30000.00,
            "bollinger_lower": 29000.00
        }
    }
    
    # Send to a test chat (this would normally be your user's chat ID)
    # You would need to set this to an actual chat ID to test
    test_chat_id = os.getenv("TEST_CHAT_ID")
    
    if test_chat_id:
        asyncio.run(bot_manager.send_signal_with_chart(test_chat_id, example_signal))
    
    # Keep the bot running
    try:
        # This is just an example, normally the bot would be part of a larger application
        input("Press Enter to stop the bot...")
    finally:
        bot_manager.stop()
