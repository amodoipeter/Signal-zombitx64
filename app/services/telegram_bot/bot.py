import logging
from aiogram import Bot, Dispatcher, types
from aiogram.types import ParseMode
from aiogram.utils import executor
import asyncio
from datetime import datetime
from typing import List, Dict, Optional
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models.signal import Signal, SignalStatus
from app.models.user import User
from app.services.telegram_bot.chart_generator import generate_chart_image
from app.services.analytics.performance import performance_analytics
from app.core.database import AsyncSessionLocal

logger = logging.getLogger(__name__)

# Initialize bot and dispatcher
bot = Bot(token=settings.TELEGRAM_BOT_TOKEN)
dp = Dispatcher(bot)

class TelegramBotService:
    def __init__(self):
        self.bot = bot
        self.dp = dp
    
    async def setup(self):
        """Set up bot commands and handlers."""
        commands = [
            types.BotCommand("/start", "Start the bot and register"),
            types.BotCommand("/help", "Get help information"),
            types.BotCommand("/signals", "Get latest signals"),
            types.BotCommand("/subscription", "Check your subscription status"),
            types.BotCommand("/weekly_report", "Get weekly performance report"),
            types.BotCommand("/monthly_report", "Get monthly performance report"),
            types.BotCommand("/redeem", "Redeem a subscription code"),  # New command
        ]
        await self.bot.set_my_commands(commands)
        
        # Register message handlers
        self._register_handlers()
    
    def _register_handlers(self):
        """Register message handlers with the dispatcher."""
        @self.dp.message_handler(commands=['start'])
        async def start_handler(message: types.Message):
            await message.answer(
                "Welcome to ZombitX64 Trading Signals Bot!\n\n"
                "I'll send you AI-powered trading signals based on your subscription level.\n"
                "Type /help to see available commands."
            )
        
        @self.dp.message_handler(commands=['help'])
        async def help_handler(message: types.Message):
            help_text = (
                "📈 <b>ZombitX64 Trading Signals Bot</b> 📉\n\n"
                "<b>Available commands:</b>\n"
                "/start - Start the bot and register\n"
                "/help - Show this help message\n"
                "/signals - Get latest signals\n"
                "/subscription - Check your subscription status\n"
                "/weekly_report - Get weekly performance report\n"
                "/monthly_report - Get monthly performance report\n"
                "/redeem - Redeem a subscription code\n\n"  # New command in help
                "Need further assistance? Contact support@zombitx64signals.com"
            )
            await message.answer(help_text, parse_mode=ParseMode.HTML)
        
        @self.dp.message_handler(commands=['weekly_report'])
        async def weekly_report_handler(message: types.Message):
            try:
                # Create a database session
                async with AsyncSessionLocal() as db:
                    # Generate weekly report
                    report_text = await performance_analytics.generate_performance_report(db, period="weekly")
                    await message.answer(report_text, parse_mode=ParseMode.HTML)
            except Exception as e:
                logger.error(f"Error generating weekly report: {str(e)}")
                await message.answer("Sorry, there was an error generating the weekly report.")
        
        @self.dp.message_handler(commands=['monthly_report'])
        async def monthly_report_handler(message: types.Message):
            try:
                # Create a database session
                async with AsyncSessionLocal() as db:
                    # Generate monthly report
                    report_text = await performance_analytics.generate_performance_report(db, period="monthly")
                    await message.answer(report_text, parse_mode=ParseMode.HTML)
            except Exception as e:
                logger.error(f"Error generating monthly report: {str(e)}")
                await message.answer("Sorry, there was an error generating the monthly report.")
        
        @self.dp.message_handler(commands=['redeem'])
        async def redeem_handler(message: types.Message):
            """Handle redeem code commands."""
            command_parts = message.text.split()
            
            # Check if code was provided with command
            if len(command_parts) > 1:
                code = command_parts[1].strip().upper()
                await process_redeem_code(message, code)
            else:
                # Ask for the code
                await message.answer(
                    "Please enter your redemption code.\n"
                    "Example: <code>ABCD-1234-EFGH</code>",
                    parse_mode=ParseMode.HTML
                )
                
                # Set up a state to wait for a code
                @self.dp.message_handler(lambda m: m.chat.id == message.chat.id, state='*')
                async def code_provided(msg: types.Message):
                    code = msg.text.strip().upper()
                    await process_redeem_code(msg, code)
        
        async def process_redeem_code(message: types.Message, code: str):
            """Process a redemption code."""
            try:
                # First find the user by their telegram_id
                async with AsyncSessionLocal() as db:
                    from sqlalchemy.future import select
                    from app.models.user import User
                    from app.services.redeem.code_service import redeem_code_service
                    
                    # Find user by telegram ID
                    result = await db.execute(
                        select(User).filter(User.telegram_chat_id == str(message.chat.id))
                    )
                    user = result.scalar_one_or_none()
                    
                    if not user:
                        await message.answer(
                            "⚠️ Your Telegram account is not linked to any user account. "
                            "Please visit our website to link your account first."
                        )
                        return
                    
                    # Try to redeem the code
                    try:
                        result = await redeem_code_service.redeem_code(
                            db=db,
                            code_str=code,
                            user_id=user.id
                        )
                        
                        # Format success message
                        sub_info = result['subscription']
                        response = (
                            "✅ <b>Code Redeemed Successfully!</b>\n\n"
                            f"<b>Subscription tier:</b> {sub_info['tier']}\n"
                            f"<b>Duration:</b> {sub_info['duration_days']} days\n"
                            f"<b>Expires:</b> {sub_info['end_date']}\n\n"
                        )
                        
                        # Add upgrade info if applicable
                        if sub_info.get('tier_upgraded'):
                            response += (
                                f"🔼 Your subscription was upgraded from "
                                f"{sub_info['old_tier']} to {sub_info['new_tier']}!\n\n"
                            )
                        
                        response += "Thank you for subscribing to ZombitX64 Trading Signals!"
                        
                        await message.answer(response, parse_mode=ParseMode.HTML)
                        
                    except ValueError as e:
                        await message.answer(f"❌ <b>Error:</b> {str(e)}", parse_mode=ParseMode.HTML)
            
            except Exception as e:
                logger.error(f"Error processing redeem code: {str(e)}")
                await message.answer(
                    "❌ There was an error processing your redemption code. "
                    "Please try again later or contact support."
                )
    
    async def send_signal(self, user: User, signal: Signal) -> bool:
        """Send a trading signal to a specific user via Telegram."""
        if not user.telegram_chat_id:
            logger.warning(f"User {user.id} doesn't have a Telegram chat ID")
            return False
        
        try:
            # Generate chart image if not available
            chart_image = None
            if not signal.chart_url:
                chart_image = await generate_chart_image(signal.symbol, signal.timeframe)
            
            # Create signal message
            emoji = "🟢" if signal.signal_type == "BUY" else "🔴"
            message = (
                f"{emoji} <b>{signal.symbol} {signal.signal_type} SIGNAL</b> {emoji}\n\n"
                f"<b>Timeframe:</b> {signal.timeframe}\n"
                f"<b>Entry price:</b> {signal.entry_price:.6f}\n"
                f"<b>Take profit:</b> {signal.take_profit:.6f}\n"
                f"<b>Stop loss:</b> {signal.stop_loss:.6f}\n"
                f"<b>Risk/Reward:</b> 1:{signal.risk_reward_ratio:.2f}\n"
                f"<b>AI Confidence:</b> {signal.confidence_score}%\n\n"
                f"<b>Analysis:</b>\n{signal.analysis_summary}\n\n"
                f"<i>Signal generated at {signal.created_at.strftime('%Y-%m-%d %H:%M:%S')} UTC</i>"
            )
            
            # Send message with chart if available
            if chart_image:
                await self.bot.send_photo(
                    chat_id=user.telegram_chat_id,
                    photo=chart_image,
                    caption=message,
                    parse_mode=ParseMode.HTML
                )
            else:
                await self.bot.send_message(
                    chat_id=user.telegram_chat_id,
                    text=message,
                    parse_mode=ParseMode.HTML
                )
            
            return True
            
        except Exception as e:
            logger.error(f"Error sending Telegram signal to user {user.id}: {str(e)}")
            return False
    
    async def send_signal_update(self, user: User, signal: Signal, status: SignalStatus) -> bool:
        """Send a signal update (TP hit, SL hit, etc.) to a user."""
        if not user.telegram_chat_id:
            return False
        
        try:
            if status == SignalStatus.TP_HIT:
                emoji = "✅"
                title = "TAKE PROFIT HIT"
            elif status == SignalStatus.SL_HIT:
                emoji = "❌"
                title = "STOP LOSS HIT"
            else:
                emoji = "ℹ️"
                title = "SIGNAL UPDATE"
            
            message = (
                f"{emoji} <b>{signal.symbol} {title}</b> {emoji}\n\n"
                f"<b>Signal type:</b> {signal.signal_type}\n"
                f"<b>Entry price:</b> {signal.entry_price:.6f}\n"
                f"<b>{'Profit' if status == SignalStatus.TP_HIT else 'Loss'}:</b> "
                f"{abs(signal.profit_loss):.2f}%\n\n"
                f"<i>Signal closed at {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC</i>"
            )
            
            await self.bot.send_message(
                chat_id=user.telegram_chat_id,
                text=message,
                parse_mode=ParseMode.HTML
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Error sending Telegram update to user {user.id}: {str(e)}")
            return False
    
    async def send_performance_report(self, users: List[User], report_text: str) -> None:
        """Send a performance report to multiple users."""
        for user in users:
            if not user.telegram_chat_id:
                continue
                
            try:
                await self.bot.send_message(
                    chat_id=user.telegram_chat_id,
                    text=report_text,
                    parse_mode=ParseMode.HTML
                )
            except Exception as e:
                logger.error(f"Error sending performance report to user {user.id}: {str(e)}")
    
    async def remove_expired_user(self, user: User) -> bool:
        """Remove a user with expired subscription from premium groups."""
        if not user.telegram_chat_id:
            logger.warning(f"User {user.id} doesn't have a Telegram chat ID")
            return False
        
        try:
            # Get all configured premium groups from settings
            premium_group_ids = settings.TELEGRAM_PREMIUM_GROUP_IDS
            
            if not premium_group_ids:
                logger.warning("No premium group IDs configured in settings")
                return False
            
            # Parse the comma-separated list of group IDs
            groups = [group_id.strip() for group_id in premium_group_ids.split(',') if group_id.strip()]
            
            # Kick user from each premium group
            for group_id in groups:
                try:
                    # Kick the user
                    await self.bot.kick_chat_member(
                        chat_id=group_id,
                        user_id=int(user.telegram_id),
                        until_date=0  # Permanent ban
                    )
                    
                    # Unban to allow them to rejoin later if they resubscribe
                    await self.bot.unban_chat_member(
                        chat_id=group_id,
                        user_id=int(user.telegram_id),
                        only_if_banned=True
                    )
                    
                    logger.info(f"Removed user {user.id} from Telegram group {group_id}")
                except Exception as e:
                    logger.error(f"Error removing user {user.id} from group {group_id}: {str(e)}")
            
            return True
        
        except Exception as e:
            logger.error(f"Error removing expired Telegram user {user.id}: {str(e)}")
            return False
    
    async def send_expiry_notification(self, user: User, message: str) -> bool:
        """Send subscription expiry notification to user."""
        if not user.telegram_chat_id:
            return False
        
        try:
            await self.bot.send_message(
                chat_id=user.telegram_chat_id,
                text=message,
                parse_mode=ParseMode.HTML
            )
            return True
        except Exception as e:
            logger.error(f"Error sending expiry notification to user {user.id}: {str(e)}")
            return False

    async def start_polling(self):
        """Start the bot polling for updates."""
        await self.setup()
        await dp.start_polling()

# Create bot service instance
telegram_bot_service = TelegramBotService()
