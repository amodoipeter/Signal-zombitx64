import discord
import asyncio
import logging
from discord.ext import commands
from typing import List, Optional, Dict
from datetime import datetime

from app.core.config import settings
from app.models.signal import Signal, SignalStatus
from app.models.user import User
from app.services.telegram_bot.chart_generator import generate_chart_image

logger = logging.getLogger(__name__)

class DiscordBotService:
    def __init__(self):
        """Initialize Discord bot with intents."""
        intents = discord.Intents.default()
        intents.message_content = True
        
        self.bot = commands.Bot(command_prefix='/', intents=intents)
        self.token = settings.DISCORD_BOT_TOKEN
        self._setup_commands()
    
    def _setup_commands(self):
        """Set up bot commands."""
        @self.bot.event
        async def on_ready():
            logger.info(f"Discord bot logged in as {self.bot.user}")
            
        @self.bot.command(name="start")
        async def start_command(ctx):
            """Start command to register user."""
            await ctx.send(
                "Welcome to ZombitX64 Trading Signals Bot!\n\n"
                "I'll send you AI-powered trading signals based on your subscription level.\n"
                "Type `/help` to see available commands."
            )
        
        @self.bot.command(name="help")
        async def help_command(ctx):
            """Help command to show available commands."""
            help_text = (
                "📈 **ZombitX64 Trading Signals Bot** 📉\n\n"
                "**Available commands:**\n"
                "`/start` - Start the bot and register\n"
                "`/help` - Show this help message\n"
                "`/signals` - Get latest signals\n"
                "`/subscription` - Check your subscription status\n\n"
                "Need further assistance? Contact support@zombitx64signals.com"
            )
            await ctx.send(help_text)
    
    async def send_signal(self, user: User, signal: Signal) -> bool:
        """Send a trading signal to a specific user via Discord."""
        if not user.discord_id:
            logger.warning(f"User {user.id} doesn't have a Discord ID")
            return False
        
        try:
            # Find the user and create a DM channel
            discord_user = await self.bot.fetch_user(int(user.discord_id))
            dm_channel = await discord_user.create_dm()
            
            # Generate chart image if not available
            chart_image = None
            if not signal.chart_url:
                chart_image = await generate_chart_image(signal.symbol, signal.timeframe)
            
            # Create signal message
            emoji = "🟢" if signal.signal_type == "BUY" else "🔴"
            message = (
                f"{emoji} **{signal.symbol} {signal.signal_type} SIGNAL** {emoji}\n\n"
                f"**Timeframe:** {signal.timeframe}\n"
                f"**Entry price:** {signal.entry_price:.6f}\n"
                f"**Take profit:** {signal.take_profit:.6f}\n"
                f"**Stop loss:** {signal.stop_loss:.6f}\n"
                f"**Risk/Reward:** 1:{signal.risk_reward_ratio:.2f}\n"
                f"**AI Confidence:** {signal.confidence_score}%\n\n"
                f"**Analysis:**\n{signal.analysis_summary}\n\n"
                f"*Signal generated at {signal.created_at.strftime('%Y-%m-%d %H:%M:%S')} UTC*"
            )
            
            # Send message with chart if available
            if chart_image:
                file = discord.File(fp=chart_image, filename=f"{signal.symbol}_{signal.timeframe}.png")
                await dm_channel.send(content=message, file=file)
            else:
                await dm_channel.send(content=message)
            
            return True
            
        except Exception as e:
            logger.error(f"Error sending Discord signal to user {user.id}: {str(e)}")
            return False
    
    async def send_signal_update(self, user: User, signal: Signal, status: SignalStatus) -> bool:
        """Send a signal update (TP hit, SL hit, etc.) to a user."""
        if not user.discord_id:
            return False
        
        try:
            # Find the user and create a DM channel
            discord_user = await self.bot.fetch_user(int(user.discord_id))
            dm_channel = await discord_user.create_dm()
            
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
                f"{emoji} **{signal.symbol} {title}** {emoji}\n\n"
                f"**Signal type:** {signal.signal_type}\n"
                f"**Entry price:** {signal.entry_price:.6f}\n"
                f"**{'Profit' if status == SignalStatus.TP_HIT else 'Loss'}:** "
                f"{abs(signal.profit_loss):.2f}%\n\n"
                f"*Signal closed at {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC*"
            )
            
            await dm_channel.send(content=message)
            
            return True
            
        except Exception as e:
            logger.error(f"Error sending Discord update to user {user.id}: {str(e)}")
            return False
    
    async def remove_expired_user(self, user: User) -> bool:
        """Remove a user with expired subscription from premium Discord channels."""
        if not user.discord_id:
            logger.warning(f"User {user.id} doesn't have a Discord ID")
            return False
        
        try:
            # Get guild and role IDs from settings
            guild_id = settings.DISCORD_GUILD_ID
            premium_role_ids = settings.DISCORD_PREMIUM_ROLE_IDS
            
            if not guild_id or not premium_role_ids:
                logger.warning("Discord guild ID or premium role IDs not configured")
                return False
            
            # Get the guild
            guild = self.bot.get_guild(int(guild_id))
            if not guild:
                logger.error(f"Guild with ID {guild_id} not found")
                return False
            
            # Get the member
            member = await guild.fetch_member(int(user.discord_id))
            if not member:
                logger.warning(f"Discord user {user.discord_id} not found in guild {guild_id}")
                return False
            
            # Parse role IDs
            role_ids = [role_id.strip() for role_id in premium_role_ids.split(',') if role_id.strip()]
            
            # Remove premium roles
            for role_id in role_ids:
                try:
                    role = guild.get_role(int(role_id))
                    if role and role in member.roles:
                        await member.remove_roles(role, reason="Subscription expired")
                        logger.info(f"Removed role {role.name} from user {user.id}")
                except Exception as e:
                    logger.error(f"Error removing role {role_id} from user {user.id}: {str(e)}")
            
            # Send expiry message
            try:
                dm_channel = await member.create_dm()
                await dm_channel.send(
                    "⚠️ **Subscription Expired** ⚠️\n\n"
                    "Your subscription has expired and your premium access has been revoked.\n\n"
                    "To continue receiving trading signals, please renew your subscription. "
                    "Visit our website or contact support for assistance.\n\n"
                    "Thank you for being part of ZombitX64 Trading Signals!"
                )
            except Exception as e:
                logger.error(f"Error sending Discord expiry message to user {user.id}: {str(e)}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error removing expired Discord user {user.id}: {str(e)}")
            return False
    
    async def start(self):
        """Start the Discord bot."""
        await self.bot.start(self.token)
    
    async def close(self):
        """Close the Discord bot connection."""
        await self.bot.close()

# Create a singleton instance
discord_bot_service = DiscordBotService()
