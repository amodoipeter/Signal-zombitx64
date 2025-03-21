import discord
from discord import Embed, Color
from typing import Dict, List, Optional
from datetime import datetime

from app.models.signal import Signal, SignalStatus, SignalType
from app.models.subscription import Subscription, SubscriptionTier

class DiscordEmbedGenerator:
    """Utility class for generating Discord embeds."""
    
    @staticmethod
    def create_signal_embed(signal: Signal) -> Embed:
        """
        Create a rich embed for a trading signal.
        
        Args:
            signal: The signal to create an embed for
            
        Returns:
            Discord Embed object
        """
        # Determine color and emoji based on signal type
        if signal.signal_type == SignalType.BUY:
            color = Color.green()
            emoji = "🟢"
        else:  # SELL
            color = Color.red()
            emoji = "🔴"
        
        # Create base embed
        embed = Embed(
            title=f"{emoji} {signal.symbol} {signal.signal_type} SIGNAL",
            description=f"**Timeframe:** {signal.timeframe}",
            color=color,
            timestamp=datetime.utcnow()
        )
        
        # Format prices with appropriate precision
        # Use more decimal places for crypto, fewer for forex
        is_crypto = 'USD' in signal.symbol or 'BTC' in signal.symbol
        precision = 6 if is_crypto else 4
        
        # Add signal details
        embed.add_field(name="Entry Price", value=f"{signal.entry_price:.{precision}f}", inline=True)
        embed.add_field(name="Take Profit", value=f"{signal.take_profit:.{precision}f}", inline=True)
        embed.add_field(name="Stop Loss", value=f"{signal.stop_loss:.{precision}f}", inline=True)
        
        # Add risk metrics
        embed.add_field(name="Risk/Reward", value=f"1:{signal.risk_reward_ratio:.2f}", inline=True)
        embed.add_field(name="Confidence", value=f"{signal.confidence_score}%", inline=True)
        
        if signal.strategy_name:
            embed.add_field(name="Strategy", value=signal.strategy_name, inline=True)
        
        # Add technical analysis if available
        if signal.analysis_summary:
            # If analysis is too long, truncate it
            analysis = signal.analysis_summary
            if len(analysis) > 1024:
                analysis = analysis[:1021] + "..."
            embed.add_field(name="Analysis", value=analysis, inline=False)
        
        # Add metadata in footer
        embed.set_footer(text=f"ZombitX64 Trading Signals • ID: {signal.id}")
        
        return embed
    
    @staticmethod
    def create_signal_update_embed(signal: Signal, status: SignalStatus) -> Embed:
        """
        Create an embed for a signal status update.
        
        Args:
            signal: The signal that was updated
            status: The new status
            
        Returns:
            Discord Embed object
        """
        # Set appropriate title and color based on status
        if status == SignalStatus.TP_HIT:
            title = f"✅ TAKE PROFIT HIT - {signal.symbol}"
            color = Color.green()
            result_text = "PROFIT"
        elif status == SignalStatus.SL_HIT:
            title = f"❌ STOP LOSS HIT - {signal.symbol}"
            color = Color.red()
            result_text = "LOSS"
        elif status == SignalStatus.EXPIRED:
            title = f"⏱️ SIGNAL EXPIRED - {signal.symbol}"
            color = Color.gold()
            result_text = "EXPIRED"
        else:
            title = f"ℹ️ SIGNAL UPDATE - {signal.symbol}"
            color = Color.blue()
            result_text = "UPDATED"
        
        # Create embed
        embed = Embed(
            title=title,
            description=f"{signal.symbol} {signal.signal_type} signal has {result_text.lower()}",
            color=color,
            timestamp=datetime.utcnow()
        )
        
        # Add signal details
        precision = 6 if 'USD' in signal.symbol or 'BTC' in signal.symbol else 4
        embed.add_field(name="Entry Price", value=f"{signal.entry_price:.{precision}f}", inline=True)
        
        if status == SignalStatus.TP_HIT:
            embed.add_field(name="Take Profit", value=f"{signal.take_profit:.{precision}f}", inline=True)
            embed.add_field(name="Profit", value=f"+{abs(signal.profit_loss):.2f}%", inline=True)
        elif status == SignalStatus.SL_HIT:
            embed.add_field(name="Stop Loss", value=f"{signal.stop_loss:.{precision}f}", inline=True)
            embed.add_field(name="Loss", value=f"-{abs(signal.profit_loss):.2f}%", inline=True)
        
        if signal.strategy_name:
            embed.add_field(name="Strategy", value=signal.strategy_name, inline=False)
        
        # Add signal duration
        if signal.created_at and signal.close_time:
            duration = signal.close_time - signal.created_at
            hours, remainder = divmod(duration.total_seconds(), 3600)
            minutes, seconds = divmod(remainder, 60)
            duration_text = f"{int(hours)}h {int(minutes)}m"
            embed.add_field(name="Signal Duration", value=duration_text, inline=True)
        
        # Add metadata in footer
        embed.set_footer(text=f"ZombitX64 Trading Signals • ID: {signal.id}")
        
        return embed
    
    @staticmethod
    def create_performance_report_embed(report_data: Dict, period: str = "weekly") -> Embed:
        """
        Create an embed for performance reports.
        
        Args:
            report_data: Dictionary containing report data
            period: "weekly" or "monthly"
            
        Returns:
            Discord Embed object
        """
        title = f"📊 {period.capitalize()} Performance Report"
        
        embed = Embed(
            title=title,
            description=f"Trading signal performance from {report_data.get('period_start', 'N/A')} to {report_data.get('period_end', 'N/A')}",
            color=Color.blue(),
            timestamp=datetime.utcnow()
        )
        
        # Performance metrics
        embed.add_field(name="Win Rate", value=f"{report_data.get('win_rate', 0):.2f}%", inline=True)
        embed.add_field(name="Total Signals", value=str(report_data.get('total_signals', 0)), inline=True)
        embed.add_field(name="Profitable", value=str(report_data.get('profitable_signals', 0)), inline=True)
        
        embed.add_field(name="Average Profit", value=f"{report_data.get('avg_profit', 0):.2f}%", inline=True)
        embed.add_field(name="Average Loss", value=f"{report_data.get('avg_loss', 0):.2f}%", inline=True)
        embed.add_field(name="Expired Signals", value=str(report_data.get('expired_signals', 0)), inline=True)
        
        # Market data
        if report_data.get('best_market') and report_data.get('best_market') != "N/A":
            embed.add_field(name="Best Market", value=report_data.get('best_market', "N/A"), inline=True)
        if report_data.get('worst_market') and report_data.get('worst_market') != "N/A":
            embed.add_field(name="Worst Market", value=report_data.get('worst_market', "N/A"), inline=True)
        
        embed.set_footer(text="ZombitX64 Trading Signals")
        
        return embed
    
    @staticmethod
    def create_subscription_embed(subscription: Subscription, is_new: bool = False) -> Embed:
        """
        Create an embed showing subscription details.
        
        Args:
            subscription: The subscription to display
            is_new: Whether this is a new subscription or renewal
            
        Returns:
            Discord Embed object
        """
        if is_new:
            title = "🎉 New Subscription Activated!"
            description = "Your subscription has been successfully activated."
        else:
            title = "📋 Subscription Status"
            description = "Your current subscription information."
        
        # Choose color based on tier
        tier_colors = {
            SubscriptionTier.FREE: Color.light_grey(),
            SubscriptionTier.BASIC: Color.blue(),
            SubscriptionTier.PREMIUM: Color.purple(),
            SubscriptionTier.VIP: Color.gold()
        }
        color = tier_colors.get(subscription.tier, Color.blue())
        
        embed = Embed(
            title=title,
            description=description,
            color=color,
            timestamp=datetime.utcnow()
        )
        
        # Add subscription details
        embed.add_field(name="Plan", value=subscription.tier, inline=True)
        embed.add_field(name="Status", value=subscription.status, inline=True)
        embed.add_field(name="Payment Method", value=subscription.payment_method, inline=True)
        
        # Add dates
        embed.add_field(name="Start Date", value=subscription.start_date.strftime("%Y-%m-%d"), inline=True)
        embed.add_field(name="Expiry Date", value=subscription.end_date.strftime("%Y-%m-%d"), inline=True)
        
        # Calculate remaining days
        days_remaining = (subscription.end_date - datetime.utcnow()).days
        embed.add_field(name="Days Remaining", value=str(max(0, days_remaining)), inline=True)
        
        # Features based on tier
        features = {
            SubscriptionTier.FREE: [
                "✅ Basic Crypto Signals",
                "✅ Daily Analysis",
                "❌ Premium Signals",
                "❌ Forex Signals",
                "❌ Priority Support"
            ],
            SubscriptionTier.BASIC: [
                "✅ All Crypto Signals",
                "✅ Daily Analysis",
                "✅ Technical Indicators",
                "❌ Forex Signals",
                "❌ Priority Support"
            ],
            SubscriptionTier.PREMIUM: [
                "✅ All Crypto Signals",
                "✅ Forex Signals",
                "✅ Advanced Analysis",
                "✅ Early Signal Access",
                "❌ VIP Strategies"
            ],
            SubscriptionTier.VIP: [
                "✅ All Crypto Signals",
                "✅ All Forex Signals",
                "✅ VIP-only Strategies",
                "✅ Priority Support",
                "✅ Custom Notifications"
            ]
        }
        
        tier_features = features.get(subscription.tier, features[SubscriptionTier.FREE])
        embed.add_field(name="Features", value="\n".join(tier_features), inline=False)
        
        # Add renewal info if close to expiry
        if days_remaining < 7 and days_remaining >= 0:
            embed.add_field(
                name="Renewal",
                value="⚠️ Your subscription is ending soon! Visit our website to renew.",
                inline=False
            )
        
        embed.set_footer(text="ZombitX64 Trading Signals")
        
        return embed

# Create singleton instance
discord_embed_generator = DiscordEmbedGenerator()
