"""
Telegram Bot Integration for the AI Signal Provider.
This module handles the sending of trading signals to subscribers via Telegram.
"""

import os
import logging
import asyncio
from datetime import datetime
from telegram import Update, ParseMode
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
        
        # Add additional information based on signal type
        if signal == "BUY":
            message += (
                f"*Recommendation:*\n"
                f"✅ Consider entering a long position\n"
                f"✅ Set stop loss at approximately `{price * 0.95:.2f}`\n"
                f"✅ Target profit at approximately `{price * 1.1:.2f}`\n"
            )
        elif signal == "SELL":
            message += (
                f"*Recommendation:*\n"
                f"✅ Consider entering a short position\n"
                f"✅ Set stop loss at approximately `{price * 1.05:.2f}`\n"
                f"✅ Target profit at approximately `{price * 0.9:.2f}`\n"
            )
        else:
            message += "*Recommendation:* ⚠️ Wait for a clearer signal\n"
        
        message += "\n_This is an automated signal and not financial advice. Always do your own research before trading._"
        
        return message
    
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
        "timestamp": datetime.now().isoformat()
    }
    
    # Send to a test chat (this would normally be your user's chat ID)
    # You would need to set this to an actual chat ID to test
    test_chat_id = os.getenv("TEST_CHAT_ID")
    
    if test_chat_id:
        asyncio.run(bot_manager.send_signal(test_chat_id, example_signal))
    
    # Keep the bot running
    try:
        # This is just an example, normally the bot would be part of a larger application
        input("Press Enter to stop the bot...")
    finally:
        bot_manager.stop()
