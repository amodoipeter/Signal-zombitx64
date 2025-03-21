import asyncio
import logging
import aiocron
from datetime import datetime, timedelta
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.future import select
from sqlalchemy.orm import sessionmaker
from typing import AsyncGenerator

from app.core.config import settings
from app.models.signal import Signal, SignalStatus, SignalType
from app.models.user import User
from app.models.subscription import Subscription, SubscriptionStatus
from app.core.ai.signal_generator import AISignalGenerator
from app.services.telegram_bot.bot import telegram_bot_service
# Discord reference kept but not used
from app.services.discord.bot import discord_bot_service
from app.services.analytics.performance import performance_analytics
from app.core.market_data.fetcher import MarketDataFetcher

logger = logging.getLogger(__name__)

# Async database connection
engine = create_async_engine(settings.DATABASE_URI.replace('postgresql://', 'postgresql+asyncpg://'))
AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

class SignalScheduler:
    def __init__(self):
        self.signal_generator = AISignalGenerator()
        self.running = False
        self.jobs = []
    async def get_db(self) -> AsyncGenerator[AsyncSession, None]:
        """Get database session."""
        async with AsyncSessionLocal() as session:
            yield session
            yield session
    
    async def start(self):
        """Start all scheduled jobs."""
        if self.running:
            return
        
        self.running = True
        
        # Schedule jobs
        self.jobs = [
            aiocron.crontab('0 */4 * * *', func=self.generate_daily_signals),  # Every 4 hours
            aiocron.crontab('*/15 * * * *', func=self.check_signal_status),    # Every 15 minutes
            aiocron.crontab('0 0 * * *', func=self.cleanup_expired_signals),   # Daily at midnight
            aiocron.crontab('0 20 * * 0', func=self.send_weekly_report),       # Sunday at 8 PM
            aiocron.crontab('0 20 1 * *', func=self.send_monthly_report),      # 1st day of month at 8 PM
            aiocron.crontab('0 1 * * *', func=self.check_expired_subscriptions),  # Daily at 1 AM
        ]
        
        logger.info("Signal scheduler started")
    
    async def stop(self):
        """Stop all scheduled jobs."""
        if not self.running:
            return
        
        # Stop all jobs
        for job in self.jobs:
            job.stop()
        
        self.running = False
        logger.info("Signal scheduler stopped")
    
    async def generate_daily_signals(self):
        """Generate signals for major pairs."""
        logger.info("Generating daily signals")
        
        try:
            # Define symbols and timeframes
            crypto_symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "DOGEUSDT"]
            forex_symbols = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "XAUUSD"]  # Added XAUUSD
            timeframes = ["1h", "4h", "1d"]
            
            # Generate signals
            all_symbols = crypto_symbols + forex_symbols
            signals = await self.signal_generator.generate_signals(all_symbols, timeframes)
            
            if not signals:
                logger.info("No signals generated")
                return
            
            logger.info(f"Generated {len(signals)} signals")
            
            # Save signals to database
            async with AsyncSessionLocal() as db:
                for signal in signals:
                    db.add(signal)
                await db.commit()
                
                # Refresh signals to get IDs
                for signal in signals:
                    await db.refresh(signal)
                
                # Get all active subscribed users with Telegram
                result = await db.execute(
                    select(User).join(Subscription).filter(
                        User.telegram_chat_id.isnot(None),
                        Subscription.status == SubscriptionStatus.ACTIVE,
                        Subscription.end_date > datetime.utcnow()
                    ).distinct()
                )
                users = result.scalars().all()
                
                # Send signals to users via Telegram only
                for signal in signals:
                    for user in users:
                        if user.telegram_chat_id:
                            await telegram_bot_service.send_signal(user, signal)
        
        except Exception as e:
            logger.error(f"Error generating signals: {str(e)}")
    
    async def check_signal_status(self):
        """Check status of active signals and update if TP/SL hit."""
        logger.info("Checking active signals status")
        
        try:
            async with AsyncSessionLocal() as db:
                # Get all active signals
                result = await db.execute(
                    select(Signal).filter(Signal.status == SignalStatus.ACTIVE)
                )
                active_signals = result.scalars().all()
                
                if not active_signals:
                    logger.info("No active signals to check")
                    return
                
                market_fetcher = MarketDataFetcher()
                
                for signal in active_signals:
                    # Get current price
                    current_price = await self._get_current_price(market_fetcher, signal.symbol)
                    
                    if current_price is None:
                        continue
                    
                    # Check if TP/SL hit
                    status_changed = False
                    
                    if signal.signal_type == SignalType.BUY:
                        if current_price >= signal.take_profit:
                            signal.status = SignalStatus.TP_HIT
                            signal.close_time = datetime.utcnow()
                            signal.profit_loss = (signal.take_profit - signal.entry_price) / signal.entry_price * 100
                            status_changed = True
                        elif current_price <= signal.stop_loss:
                            signal.status = SignalStatus.SL_HIT
                            signal.close_time = datetime.utcnow()
                            signal.profit_loss = (signal.stop_loss - signal.entry_price) / signal.entry_price * 100
                            status_changed = True
                    else:  # SELL
                        if current_price <= signal.take_profit:
                            signal.status = SignalStatus.TP_HIT
                            signal.close_time = datetime.utcnow()
                            signal.profit_loss = (signal.entry_price - signal.take_profit) / signal.entry_price * 100
                            status_changed = True
                        elif current_price >= signal.stop_loss:
                            signal.status = SignalStatus.SL_HIT
                            signal.close_time = datetime.utcnow()
                            signal.profit_loss = (signal.entry_price - signal.stop_loss) / signal.entry_price * 100
                            status_changed = True
                    
                    if status_changed:
                        # Update signal in DB
                        db.add(signal)
                        await db.commit()
                        await db.refresh(signal)
                        
                        # Get all subscribed users with Telegram
                        user_result = await db.execute(
                            select(User).join(Subscription).filter(
                                User.telegram_chat_id.isnot(None),
                                Subscription.status == SubscriptionStatus.ACTIVE,
                                Subscription.end_date > datetime.utcnow()
                            ).distinct()
                        )
                        users = user_result.scalars().all()
                        
                        # Send updates to users via Telegram only
                        for user in users:
                            if user.telegram_chat_id:
                                await telegram_bot_service.send_signal_update(user, signal, signal.status)
                
                # Close market fetcher connection
                await market_fetcher.close()
        
        except Exception as e:
            logger.error(f"Error checking signal status: {str(e)}")
    
    async def cleanup_expired_signals(self):
        """Mark old signals as expired."""
        logger.info("Cleaning up expired signals")
        
        try:
            # Signals older than 7 days that didn't hit TP or SL
            expiry_date = datetime.utcnow() - timedelta(days=7)
            
            async with AsyncSessionLocal() as db:
                result = await db.execute(
                    select(Signal).filter(
                        Signal.status == SignalStatus.ACTIVE,
                        Signal.created_at < expiry_date
                    )
                )
                expired_signals = result.scalars().all()
                
                if not expired_signals:
                    logger.info("No expired signals to clean up")
                    return
                
                for signal in expired_signals:
                    signal.status = SignalStatus.EXPIRED
                    signal.close_time = datetime.utcnow()
                
                await db.commit()
                logger.info(f"Marked {len(expired_signals)} signals as expired")
        
        except Exception as e:
            logger.error(f"Error cleaning up expired signals: {str(e)}")
    
    async def _get_current_price(self, market_fetcher, symbol):
        """Get current price for a symbol."""
        try:
            # Get recent OHLCV data
            df = await market_fetcher.fetch_data(symbol, "1m", limit=1)
            
            if df.empty:
                logger.warning(f"No price data available for {symbol}")
                return None
            
            # Return the closing price of the most recent candle
            return df['close'].iloc[-0]
        
        except Exception as e:
            logger.error(f"Error fetching current price for {symbol}: {str(e)}")
            return None
    
    async def send_weekly_report(self):
        """Generate and send weekly performance report to subscribers."""
        logger.info("Generating weekly performance report")
        
        try:
            async with AsyncSessionLocal() as db:
                # Generate report
                report_text = await performance_analytics.generate_performance_report(db, period="weekly")
                
                # Get all active subscribers
                result = await db.execute(
                    select(User).join(Subscription).filter(
                        Subscription.status == SubscriptionStatus.ACTIVE,
                        Subscription.end_date > datetime.utcnow(),
                        User.telegram_chat_id.isnot(None),
                    ).distinct()
                )
                users = result.scalars().all()
                
                # Send report to users
                await telegram_bot_service.send_performance_report(users, report_text)
                logger.info(f"Weekly report sent to {len(users)} users")
                
        except Exception as e:
            logger.error(f"Error sending weekly report: {str(e)}")
    
    async def send_monthly_report(self):
        """Generate and send monthly performance report to subscribers."""
        logger.info("Generating monthly performance report")
        
        try:
            async with AsyncSessionLocal() as db:
                # Generate report
                report_text = await performance_analytics.generate_performance_report(db, period="monthly")
                
                # Get all active subscribers
                result = await db.execute(
                    select(User).join(Subscription).filter(
                        Subscription.status == SubscriptionStatus.ACTIVE,
                        Subscription.end_date > datetime.utcnow(),
                        User.telegram_chat_id.isnot(None),
                    ).distinct()
                )
                users = result.scalars().all()
                
                # Send report to users
                await telegram_bot_service.send_performance_report(users, report_text)
                logger.info(f"Monthly report sent to {len(users)} users")
                
        except Exception as e:
            logger.error(f"Error sending monthly report: {str(e)}")
    
    async def check_expired_subscriptions(self):
        """Check for expired subscriptions and remove users from groups."""
        logger.info("Checking expired subscriptions")
        
        try:
            now = datetime.utcnow()
            yesterday = now - timedelta(days=1)
            
            async with AsyncSessionLocal() as db:
                # Find subscriptions that expired in the last 24 hours
                # (We use a window to avoid checking all expired subscriptions every time)
                result = await db.execute(
                    select(Subscription).filter(
                        Subscription.status == SubscriptionStatus.ACTIVE,
                        Subscription.end_date < now,
                        Subscription.end_date > yesterday
                    )
                )
                expired_subscriptions = result.scalars().all()
                
                if not expired_subscriptions:
                    logger.info("No newly expired subscriptions found")
                    return
                
                logger.info(f"Found {len(expired_subscriptions)} newly expired subscriptions")
                
                # Process each expired subscription
                for subscription in expired_subscriptions:
                    # Update subscription status
                    subscription.status = SubscriptionStatus.EXPIRED
                    
                    # Get user info
                    result = await db.execute(
                        select(User).filter(User.id == subscription.user_id)
                    )
                    user = result.scalar_one_or_none()
                    
                    if not user:
                        logger.warning(f"User not found for subscription {subscription.id}")
                        continue
                    
                    # Check if user has other active subscriptions
                    result = await db.execute(
                        select(Subscription).filter(
                            Subscription.user_id == user.id,
                            Subscription.status == SubscriptionStatus.ACTIVE,
                            Subscription.end_date > now
                        )
                    )
                    other_active_subs = result.scalars().all()
                    
                    # If no other active subscriptions, remove from groups
                    if not other_active_subs:
                        # Remove from Telegram groups only
                        if user.telegram_chat_id:
                            await telegram_bot_service.remove_expired_user(user)
                            
                        # Send notification about subscription expiry
                        await self._send_expiry_notification(user)
                
                # Commit all changes
                await db.commit()
                
        except Exception as e:
            logger.error(f"Error checking expired subscriptions: {str(e)}")
    
    async def _send_expiry_notification(self, user: User):
        """Send notification to user about subscription expiry."""
        try:
            message = (
                "⚠️ <b>Subscription Expired</b> ⚠️\n\n"
                "Your subscription has expired and you've been removed from premium groups.\n\n"
                "To continue receiving trading signals, please renew your subscription. "
                "Visit our website or contact support for assistance.\n\n"
                "Thank you for being part of ZombitX64 Trading Signals!"
            )
            
            # Only send via Telegram
            if user.telegram_chat_id:
                await telegram_bot_service.send_expiry_notification(user, message)
                
        except Exception as e:
            logger.error(f"Error sending expiry notification: {str(e)}")

signal_scheduler = SignalScheduler()
