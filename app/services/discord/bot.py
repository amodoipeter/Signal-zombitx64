import discord
import asyncio
import logging
import io
from discord.ext import commands, tasks
from discord import File, Embed, Color
from typing import List, Optional, Dict, Union
from datetime import datetime, timedelta
import os

from app.core.config import settings
from app.models.signal import Signal, SignalStatus, SignalType
from app.models.user import User
from app.services.telegram_bot.chart_generator import generate_chart_image

logger = logging.getLogger(__name__)

class DiscordBotService:
    def __init__(self):
        """Initialize Discord bot with intents."""
        intents = discord.Intents.default()
        intents.message_content = True
        intents.members = True  # Required for modifying user roles
        
        self.bot = commands.Bot(command_prefix='/', intents=intents)
        self.token = settings.DISCORD_BOT_TOKEN
        self.guild_id = settings.DISCORD_GUILD_ID if hasattr(settings, 'DISCORD_GUILD_ID') else None
        self.premium_role_ids = self._parse_role_ids(settings.DISCORD_PREMIUM_ROLE_IDS if hasattr(settings, 'DISCORD_PREMIUM_ROLE_IDS') else "")
        self.free_role_id = settings.DISCORD_FREE_ROLE_ID if hasattr(settings, 'DISCORD_FREE_ROLE_ID') else None
        self.signal_channel_id = settings.DISCORD_SIGNAL_CHANNEL_ID if hasattr(settings, 'DISCORD_SIGNAL_CHANNEL_ID') else None
        self.admin_channel_id = settings.DISCORD_ADMIN_CHANNEL_ID if hasattr(settings, 'DISCORD_ADMIN_CHANNEL_ID') else None
        self._setup_commands()
        self._running = False

    def _parse_role_ids(self, role_ids_str):
        """Parse comma-separated role IDs into a list of integers."""
        if not role_ids_str:
            return []
        return [int(role_id.strip()) for role_id in role_ids_str.split(',') if role_id.strip().isdigit()]
    
    def _setup_commands(self):
        """Set up bot commands."""
        @self.bot.event
        async def on_ready():
            logger.info(f"Discord bot logged in as {self.bot.user}")
            await self.bot.change_presence(activity=discord.Activity(
                type=discord.ActivityType.watching,
                name="for trading signals | /help"
            ))
            # Start background tasks
            if not self.check_expired_subscriptions.is_running():
                self.check_expired_subscriptions.start()
            
        @self.bot.command(name="start")
        async def start_command(ctx):
            """Start command to register user."""
            embed = Embed(
                title="Welcome to ZombitX64 Trading Signals!",
                description="I'll send you AI-powered trading signals based on your subscription level.",
                color=Color.blue()
            )
            embed.add_field(name="Get Started", value="Type `/help` to see available commands", inline=False)
            embed.add_field(name="Connect Account", value="Use `/connect your_email@example.com` to link your Discord account with our platform", inline=False)
            embed.add_field(name="Subscription", value="Check your subscription status with `/status`", inline=False)
            embed.set_footer(text="ZombitX64 Trading Signals")
            
            await ctx.send(embed=embed)
        
        @self.bot.command(name="help")
        async def help_command(ctx):
            """Help command to show available commands."""
            embed = Embed(
                title="📈 ZombitX64 Trading Signals Bot 📉",
                description="Your AI-powered trading companion",
                color=Color.blue()
            )
            
            commands_info = [
                ("`/start`", "Start the bot and get welcome information"),
                ("`/help`", "Show this help message"),
                ("`/connect email`", "Connect your Discord account to the platform"),
                ("`/signals`", "Get latest trading signals"),
                ("`/status`", "Check your subscription status"),
                ("`/report weekly`", "Get weekly performance report"),
                ("`/report monthly`", "Get monthly performance report"),
                ("`/redeem CODE`", "Redeem a subscription code")
            ]
            
            for cmd, desc in commands_info:
                embed.add_field(name=cmd, value=desc, inline=False)
            
            embed.add_field(
                name="Need Help?",
                value="Contact support@zombitx64signals.com or join our [Support Server](https://discord.gg/zombitx64)",
                inline=False
            )
            embed.set_footer(text="ZombitX64 Trading Signals")
            
            await ctx.send(embed=embed)

        @self.bot.command(name="connect")
        async def connect_command(ctx, email: str = None):
            """Connect Discord account to platform."""
            if not email:
                await ctx.send("Please provide your email address: `/connect your_email@example.com`")
                return
            
            # Here we would call an API endpoint to link the Discord account
            # For now, we'll just simulate the response
            embed = Embed(
                title="Account Connection Request",
                description=f"We've received your request to connect Discord account to {email}",
                color=Color.green()
            )
            embed.add_field(
                name="Next Steps", 
                value="We'll process your request and update your account. You'll receive a confirmation message when completed.", 
                inline=False
            )
            embed.add_field(
                name="User ID", 
                value=f"`{ctx.author.id}`", 
                inline=True
            )
            embed.set_footer(text="ZombitX64 Trading Signals")
            
            await ctx.send(embed=embed)
            
            # Send notification to admin channel if configured
            if self.admin_channel_id:
                try:
                    admin_channel = self.bot.get_channel(int(self.admin_channel_id))
                    if admin_channel:
                        admin_embed = Embed(
                            title="Discord Connect Request",
                            description=f"User requested Discord account connection",
                            color=Color.blue()
                        )
                        admin_embed.add_field(name="Email", value=email, inline=True)
                        admin_embed.add_field(name="Discord User", value=f"{ctx.author} ({ctx.author.id})", inline=True)
                        admin_embed.set_footer(text=f"Requested at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                        
                        await admin_channel.send(embed=admin_embed)
                except Exception as e:
                    logger.error(f"Error sending admin notification: {str(e)}")
        
        @self.bot.command(name="signals")
        async def signals_command(ctx):
            """Get latest trading signals."""
            # In a real implementation, this would fetch signals from the database
            # For now, send a placeholder response
            embed = Embed(
                title="Recent Trading Signals",
                description="Here are your latest trading signals",
                color=Color.gold()
            )
            
            # Simulated signals
            signals_data = [
                {"symbol": "BTCUSDT", "type": "BUY", "entry": "27850.00", "tp": "28500.00", "sl": "27500.00"},
                {"symbol": "ETHUSDT", "type": "SELL", "entry": "1680.50", "tp": "1650.00", "sl": "1700.00"}
            ]
            
            for signal in signals_data:
                signal_emoji = "🟢" if signal["type"] == "BUY" else "🔴"
                embed.add_field(
                    name=f"{signal_emoji} {signal['symbol']} {signal['type']}",
                    value=f"Entry: {signal['entry']}\nTP: {signal['tp']}\nSL: {signal['sl']}",
                    inline=True
                )
            
            embed.set_footer(text=f"Generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            await ctx.send(embed=embed)
        
        @self.bot.command(name="status")
        async def status_command(ctx):
            """Check subscription status."""
            # This would fetch actual subscription data from the database
            embed = Embed(
                title="Subscription Status",
                description="Here's your current subscription information",
                color=Color.green()
            )
            
            # Simulated subscription info
            embed.add_field(name="Plan", value="Premium", inline=True)
            embed.add_field(name="Status", value="Active", inline=True)
            embed.add_field(name="Expires", value=(datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d"), inline=True)
            
            embed.add_field(
                name="Features", 
                value="✅ Crypto Signals\n✅ Forex Signals\n✅ Premium Analysis\n✅ Priority Support",
                inline=False
            )
            
            embed.add_field(
                name="Upgrade",
                value="Visit [our website](https://zombitx64signals.com/upgrade) to upgrade your plan",
                inline=False
            )
            
            embed.set_footer(text="ZombitX64 Trading Signals")
            
            await ctx.send(embed=embed)
        
        @self.bot.command(name="report")
        async def report_command(ctx, period: str = None):
            """Get performance report."""
            if not period or period.lower() not in ["weekly", "monthly"]:
                await ctx.send("Please specify a valid period: `/report weekly` or `/report monthly`")
                return
            
            period = period.lower()
            
            embed = Embed(
                title=f"📊 {period.capitalize()} Performance Report",
                description=f"Trading performance over the past {period.rstrip('ly')}",
                color=Color.blue()
            )
            
            # Simulated performance data
            embed.add_field(name="Win Rate", value="67.5%", inline=True)
            embed.add_field(name="Total Signals", value="40", inline=True)
            embed.add_field(name="Profitable", value="27", inline=True)
            
            embed.add_field(name="Average Profit", value="2.3%", inline=True)
            embed.add_field(name="Average Loss", value="1.1%", inline=True)
            embed.add_field(name="Net Profit", value="38.2%", inline=True)
            
            embed.add_field(
                name="Best Performing Pair", 
                value="BTCUSDT (+5.2%)",
                inline=False
            )
            
            embed.set_footer(text=f"Report generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            await ctx.send(embed=embed)
        
        @self.bot.command(name="redeem")
        async def redeem_command(ctx, code: str = None):
            """Redeem a subscription code."""
            if not code:
                await ctx.send("Please provide a redeem code: `/redeem YOUR_CODE`")
                return
            
            # This would validate the code against the database
            embed = Embed(
                title="Code Redemption",
                description="Processing your code...",
                color=Color.gold()
            )
            
            # Simulated response
            embed.add_field(name="Status", value="Success", inline=True)
            embed.add_field(name="Code", value=code, inline=True)
            embed.add_field(name="Duration", value="30 days", inline=True)
            
            embed.add_field(name="Subscription", value="Your Premium subscription has been activated!", inline=False)
            embed.add_field(name="Expires", value=(datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d"), inline=True)
            
            embed.set_footer(text="Thank you for subscribing to ZombitX64 Trading Signals")
            
            await ctx.send(embed=embed)

        @tasks.loop(hours=12)  # Check twice a day
        async def check_expired_subscriptions():
            """Background task to check for expired subscriptions."""
            logger.info("Discord bot checking for expired subscriptions")
            # In a real implementation, this would query the database for expired subscriptions
            # and remove roles from users with expired subscriptions
    
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
            
            # Create signal embed
            embed = self._create_signal_embed(signal)
            
            # Send message with chart if available
            if chart_image:
                file = File(fp=chart_image, filename=f"{signal.symbol}_{signal.timeframe}.png")
                embed.set_image(url=f"attachment://{signal.symbol}_{signal.timeframe}.png")
                await dm_channel.send(embed=embed, file=file)
            else:
                await dm_channel.send(embed=embed)
            
            return True
            
        except Exception as e:
            logger.error(f"Error sending Discord signal to user {user.id}: {str(e)}")
            return False
    
    def _create_signal_embed(self, signal: Signal) -> Embed:
        """Create an embed for a trading signal."""
        # Determine color based on signal type
        color = Color.green() if signal.signal_type == SignalType.BUY else Color.red()
        
        # Determine emoji based on signal type
        emoji = "🟢" if signal.signal_type == SignalType.BUY else "🔴"
        
        # Create embed
        embed = Embed(
            title=f"{emoji} {signal.symbol} {signal.signal_type} SIGNAL",
            description=f"**Timeframe:** {signal.timeframe}",
            color=color,
            timestamp=datetime.utcnow()
        )
        
        # Add signal details
        embed.add_field(name="Entry Price", value=f"{signal.entry_price:.6f}", inline=True)
        embed.add_field(name="Take Profit", value=f"{signal.take_profit:.6f}", inline=True)
        embed.add_field(name="Stop Loss", value=f"{signal.stop_loss:.6f}", inline=True)
        
        embed.add_field(name="Risk/Reward", value=f"1:{signal.risk_reward_ratio:.2f}", inline=True)
        embed.add_field(name="Confidence", value=f"{signal.confidence_score}%", inline=True)
        embed.add_field(name="Strategy", value=f"{signal.strategy_name}", inline=True)
        
        # Add analysis
        if signal.analysis_summary:
            embed.add_field(name="Analysis", value=signal.analysis_summary[:1024], inline=False)
        
        embed.set_footer(text="ZombitX64 Trading Signals")
        
        return embed
    
    async def send_signal_update(self, user: User, signal: Signal, status: SignalStatus) -> bool:
        """Send a signal update (TP hit, SL hit, etc.) to a user."""
        if not user.discord_id:
            return False
        
        try:
            # Find the user and create a DM channel
            discord_user = await self.bot.fetch_user(int(user.discord_id))
            dm_channel = await discord_user.create_dm()
            
            # Create embed for update
            embed = self._create_signal_update_embed(signal, status)
            
            # Send the update
            await dm_channel.send(embed=embed)
            
            return True
            
        except Exception as e:
            logger.error(f"Error sending Discord update to user {user.id}: {str(e)}")
            return False
    
    def _create_signal_update_embed(self, signal: Signal, status: SignalStatus) -> Embed:
        """Create an embed for a signal update."""
        if status == SignalStatus.TP_HIT:
            title = f"✅ TAKE PROFIT HIT - {signal.symbol}"
            color = Color.green()
            description = f"**{signal.symbol} {signal.signal_type} signal hit take profit!**"
        elif status == SignalStatus.SL_HIT:
            title = f"❌ STOP LOSS HIT - {signal.symbol}"
            color = Color.red()
            description = f"**{signal.symbol} {signal.signal_type} signal hit stop loss**"
        else:
            title = f"ℹ️ SIGNAL UPDATE - {signal.symbol}"
            color = Color.gold()
            description = f"**{signal.symbol} {signal.signal_type} signal has been updated**"
        
        embed = Embed(
            title=title,
            description=description,
            color=color,
            timestamp=datetime.utcnow()
        )
        
        # Add signal details
        embed.add_field(name="Entry Price", value=f"{signal.entry_price:.6f}", inline=True)
        
        if status == SignalStatus.TP_HIT:
            embed.add_field(name="Take Profit", value=f"{signal.take_profit:.6f}", inline=True)
            embed.add_field(name="Profit", value=f"{abs(signal.profit_loss):.2f}%", inline=True)
        elif status == SignalStatus.SL_HIT:
            embed.add_field(name="Stop Loss", value=f"{signal.stop_loss:.6f}", inline=True)
            embed.add_field(name="Loss", value=f"{abs(signal.profit_loss)::.2f}%", inline=True)
        
        embed.set_footer(text="ZombitX64 Trading Signals")
        
        return embed
    
    async def send_performance_report(self, users: List[User], report_text: str) -> int:
        """Send performance report to multiple users."""
        success_count = 0
        report_lines = report_text.strip().split('\n')
        
        # Extract basic info from report_text
        title = report_lines[0] if report_lines else "📊 Performance Report"
        
        embed = Embed(
            title=title,
            description="Trading signal performance analysis",
            color=Color.blue(),
            timestamp=datetime.utcnow()
        )
        
        # Parse report content into fields
        current_field = None
        current_value = []
        
        for line in report_lines[1:]:
            line = line.strip()
            if not line:
                if current_field:
                    embed.add_field(name=current_field, value="\n".join(current_value), inline=False)
                    current_field = None
                    current_value = []
                continue
            
            if ': ' in line and not current_field:
                parts = line.split(': ', 1)
                embed.add_field(name=parts[0], value=parts[1], inline=True)
            else:
                if not current_field:
                    current_field = "Analysis"
                current_value.append(line)
        
        # Add any remaining field
        if current_field and current_value:
            embed.add_field(name=current_field, value="\n".join(current_value), inline=False)
        
        embed.set_footer(text="ZombitX64 Trading Signals")
        
        # Send to all users
        for user in users:
            if user.discord_id:
                try:
                    discord_user = await self.bot.fetch_user(int(user.discord_id))
                    dm_channel = await discord_user.create_dm()
                    await dm_channel.send(embed=embed)
                    success_count += 1
                except Exception as e:
                    logger.error(f"Error sending report to Discord user {user.id}: {str(e)}")
        
        return success_count
    
    async def send_to_channel(self, channel_id: Union[int, str], message: str = None, embed: Embed = None, file: File = None) -> bool:
        """Send a message to a specific channel."""
        try:
            channel = self.bot.get_channel(int(channel_id))
            if not channel:
                logger.error(f"Channel with ID {channel_id} not found")
                return False
            
            await channel.send(content=message, embed=embed, file=file)
            return True
        except Exception as e:
            logger.error(f"Error sending message to channel {channel_id}: {str(e)}")
            return False
    
    async def broadcast_signal(self, signal: Signal, channel_id: Union[int, str] = None) -> bool:
        """Broadcast a signal to a channel."""
        # Use the specified channel ID or the default signal channel
        target_channel_id = channel_id or self.signal_channel_id
        if not target_channel_id:
            logger.warning("No channel ID specified for signal broadcast")
            return False
        
        try:
            # Generate chart image
            chart_image = await generate_chart_image(signal.symbol, signal.timeframe)
            
            # Create signal embed
            embed = self._create_signal_embed(signal)
            
            # Create file from chart image if available
            file = None
            if chart_image:
                file = File(fp=chart_image, filename=f"{signal.symbol}_{signal.timeframe}.png")
                embed.set_image(url=f"attachment://{signal.symbol}_{signal.timeframe}.png")
            
            # Send to channel
            return await self.send_to_channel(target_channel_id, embed=embed, file=file)
            
        except Exception as e:
            logger.error(f"Error broadcasting signal to channel {target_channel_id}: {str(e)}")
            return False
    
    async def send_expiry_notification(self, user: User, message: str) -> bool:
        """Send subscription expiry notification to a user."""
        if not user.discord_id:
            return False
        
        try:
            discord_user = await self.bot.fetch_user(int(user.discord_id))
            dm_channel = await discord_user.create_dm()
            
            embed = Embed(
                title="⚠️ Subscription Expired",
                description="Your subscription has expired",
                color=Color.orange(),
                timestamp=datetime.utcnow()
            )
            
            embed.add_field(name="Important", value=message, inline=False)
            
            embed.add_field(
                name="Renew Now", 
                value="Renew your subscription at [our website](https://zombitx64signals.com/renew) to continue receiving signals",
                inline=False
            )
            
            embed.set_footer(text="ZombitX64 Trading Signals")
            
            await dm_channel.send(embed=embed)
            return True
            
        except Exception as e:
            logger.error(f"Error sending expiry notification to Discord user {user.id}: {str(e)}")
            return False
    
    async def remove_expired_user(self, user: User) -> bool:
        """Remove a user with expired subscription from premium Discord channels."""
        if not user.discord_id or not self.guild_id or not self.premium_role_ids:
            logger.warning(f"Missing required data to remove user {user.id} (Discord ID: {user.discord_id})")
            return False
        
        try:
            # Get the guild
            guild = self.bot.get_guild(int(self.guild_id))
            if not guild:
                logger.error(f"Guild with ID {self.guild_id} not found")
                return False
            
            # Get the member
            member = await guild.fetch_member(int(user.discord_id))
            if not member:
                logger.warning(f"Discord user {user.discord_id} not found in guild {self.guild_id}")
                return False
            
            # Remove premium roles
            roles_removed = 0
            for role_id in self.premium_role_ids:
                try:
                    role = guild.get_role(int(role_id))
                    if role and role in member.roles:
                        await member.remove_roles(role, reason="Subscription expired")
                        roles_removed += 1
                        logger.info(f"Removed role {role.name} from user {user.id}")
                except Exception as e:
                    logger.error(f"Error removing role {role_id} from user {user.id}: {str(e)}")
            
            # Add free role if configured
            if self.free_role_id:
                try:
                    free_role = guild.get_role(int(self.free_role_id))
                    if free_role and free_role not in member.roles:
                        await member.add_roles(free_role, reason="Subscription expired - moved to free tier")
                        logger.info(f"Added free role {free_role.name} to user {user.id}")
                except Exception as e:
                    logger.error(f"Error adding free role {self.free_role_id} to user {user.id}: {str(e)}")
            
            # Send expiry message if roles were removed
            if roles_removed > 0:
                try:
                    await self.send_expiry_notification(
                        user,
                        "Your subscription has expired and your premium access has been revoked. "
                        "To continue receiving trading signals, please renew your subscription."
                    )
                except Exception as e:
                    logger.error(f"Error sending Discord expiry message to user {user.id}: {str(e)}")
            
            return roles_removed > 0
            
        except Exception as e:
            logger.error(f"Error removing expired Discord user {user.id}: {str(e)}")
            return False
    
    async def start(self):
        """Start the Discord bot."""
        if self._running:
            return
        
        self._running = True
        logger.info("Starting Discord bot")
        
        try:
            await self.bot.start(self.token)
        except Exception as e:
            self._running = False
            logger.error(f"Error starting Discord bot: {str(e)}")
        finally:
            self._running = False
    
    async def close(self):
        """Close the Discord bot connection."""
        if not self._running:
            return
        
        logger.info("Closing Discord bot")
        await self.bot.close()
        self._running = False

# Create a singleton instance
discord_bot_service = DiscordBotService()
