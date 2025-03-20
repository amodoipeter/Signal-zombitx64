import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sqlalchemy import func, and_, select
from typing import Dict, List, Tuple, Optional
import logging

from app.models.signal import Signal, SignalStatus, SignalType
from app.models.subscription import SubscriptionTier

logger = logging.getLogger(__name__)

class PerformanceAnalytics:
    """Analytics service for calculating signal performance metrics."""
    
    @staticmethod
    async def calculate_win_rate(db, start_date: datetime, end_date: datetime, market: Optional[str] = None) -> Dict:
        """Calculate win rate statistics for a date range."""
        query = select(Signal).filter(
            and_(
                Signal.created_at >= start_date,
                Signal.created_at <= end_date,
                Signal.status.in_([SignalStatus.TP_HIT, SignalStatus.SL_HIT, SignalStatus.EXPIRED])
            )
        )
        
        if market:
            query = query.filter(Signal.market == market)
            
        result = await db.execute(query)
        signals = result.scalars().all()
        
        if not signals:
            return {
                "period_start": start_date.strftime("%Y-%m-%d"),
                "period_end": end_date.strftime("%Y-%m-%d"),
                "total_signals": 0,
                "win_rate": 0,
                "profitable_signals": 0,
                "losing_signals": 0,
                "expired_signals": 0,
                "avg_profit": 0,
                "avg_loss": 0,
                "best_market": "N/A",
                "worst_market": "N/A",
                "market": market or "all"
            }
        
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame([
            {
                "id": signal.id,
                "symbol": signal.symbol,
                "market": signal.market,
                "signal_type": signal.signal_type,
                "status": signal.status,
                "profit_loss": signal.profit_loss or 0,
                "created_at": signal.created_at
            }
            for signal in signals
        ])
        
        # Calculate statistics
        total_signals = len(df)
        profitable_signals = len(df[df["status"] == SignalStatus.TP_HIT])
        losing_signals = len(df[df["status"] == SignalStatus.SL_HIT])
        expired_signals = len(df[df["status"] == SignalStatus.EXPIRED])
        
        win_rate = (profitable_signals / total_signals) * 100 if total_signals > 0 else 0
        
        # Calculate average profit/loss
        profits = df[df["status"] == SignalStatus.TP_HIT]["profit_loss"]
        losses = df[df["status"] == SignalStatus.SL_HIT]["profit_loss"]
        
        avg_profit = profits.mean() if len(profits) > 0 else 0
        avg_loss = losses.mean() if len(losses) > 0 else 0
        
        # Find best and worst markets if analyzing all markets
        best_market = "N/A"
        worst_market = "N/A"
        
        if not market and "market" in df.columns:
            market_stats = df.groupby("market").apply(
                lambda x: (len(x[x["status"] == SignalStatus.TP_HIT]) / len(x)) * 100 if len(x) > 0 else 0
            ).sort_values(ascending=False)
            
            if not market_stats.empty:
                best_market = market_stats.index[0] if len(market_stats) > 0 else "N/A"
                worst_market = market_stats.index[-1] if len(market_stats) > 1 else "N/A"
        
        return {
            "period_start": start_date.strftime("%Y-%m-%d"),
            "period_end": end_date.strftime("%Y-%m-%d"),
            "total_signals": total_signals,
            "win_rate": round(win_rate, 2),
            "profitable_signals": profitable_signals,
            "losing_signals": losing_signals,
            "expired_signals": expired_signals,
            "avg_profit": round(avg_profit, 2),
            "avg_loss": round(avg_loss, 2),
            "best_market": best_market,
            "worst_market": worst_market,
            "market": market or "all"
        }
    
    @staticmethod
    async def get_weekly_report(db) -> Dict:
        """Get win rate report for the past week."""
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=7)
        
        return await PerformanceAnalytics.calculate_win_rate(db, start_date, end_date)
    
    @staticmethod
    async def get_monthly_report(db) -> Dict:
        """Get win rate report for the past month."""
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=30)
        
        return await PerformanceAnalytics.calculate_win_rate(db, start_date, end_date)
    
    @staticmethod
    async def generate_performance_report(db, period: str = "weekly") -> str:
        """Generate a formatted performance report."""
        if period == "weekly":
            stats = await PerformanceAnalytics.get_weekly_report(db)
            title = "📊 Weekly Performance Report 📊"
        else:  # monthly
            stats = await PerformanceAnalytics.get_monthly_report(db)
            title = "📊 Monthly Performance Report 📊"
        
        if stats["total_signals"] == 0:
            return f"{title}\n\nNo signals were generated during this period."
            
        # Build the report text
        report = f"{title}\n\n"
        report += f"📅 Period: {stats['period_start']} to {stats['period_end']}\n"
        report += f"📈 Win Rate: {stats['win_rate']}%\n"
        report += f"🎯 Total Signals: {stats['total_signals']}\n"
        report += f"✅ Profitable Signals: {stats['profitable_signals']}\n"
        report += f"❌ Losing Signals: {stats['losing_signals']}\n"
        report += f"⏳ Expired Signals: {stats['expired_signals']}\n\n"
        
        report += f"💰 Average Profit: {stats['avg_profit']}%\n"
        report += f"💸 Average Loss: {stats['avg_loss']}%\n\n"
        
        if stats["best_market"] != "N/A":
            report += f"🥇 Best Performing Market: {stats['best_market']}\n"
        if stats["worst_market"] != "N/A":
            report += f"🥉 Worst Performing Market: {stats['worst_market']}\n"
        
        return report

performance_analytics = PerformanceAnalytics()
