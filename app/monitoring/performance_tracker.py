"""
Performance Tracking module for the AI Signal Provider.
This module monitors and evaluates the performance of trading signals.
"""

import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from dotenv import load_dotenv

# Configure logging
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class PerformanceTracker:
    """
    A class to track and evaluate the performance of trading signals.
    """
    
    def __init__(self, db_connection=None):
        """
        Initialize the PerformanceTracker.
        
        Args:
            db_connection: Database connection object (optional)
        """
        self.db_connection = db_connection
        logger.info("Performance tracker initialized")
    
    def record_signal(self, signal_data):
        """
        Record a new trading signal for performance tracking.
        
        Args:
            signal_data (dict): Dictionary containing signal information
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Here you would insert the signal data into your database
            # For now, we'll just log it
            logger.info(f"Recorded signal: {signal_data}")
            return True
        except Exception as e:
            logger.error(f"Error recording signal: {str(e)}")
            return False
    
    def record_outcome(self, signal_id, outcome_data):
        """
        Record the outcome of a previously issued trading signal.
        
        Args:
            signal_id (str): ID of the signal
            outcome_data (dict): Dictionary containing outcome information
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Here you would update the signal record in your database
            # For now, we'll just log it
            logger.info(f"Recorded outcome for signal {signal_id}: {outcome_data}")
            return True
        except Exception as e:
            logger.error(f"Error recording outcome: {str(e)}")
            return False
    
    def calculate_performance_metrics(self, signals, timeframe="all"):
        """
        Calculate performance metrics for a set of signals.
        
        Args:
            signals (list): List of signal dictionaries with outcomes
            timeframe (str): Time period to calculate metrics for ('all', 'week', 'month', 'year')
            
        Returns:
            dict: Dictionary of performance metrics
        """
        # Filter signals by timeframe if needed
        if timeframe != "all":
            now = datetime.now()
            if timeframe == "week":
                cutoff = now - timedelta(days=7)
            elif timeframe == "month":
                cutoff = now - timedelta(days=30)
            elif timeframe == "year":
                cutoff = now - timedelta(days=365)
            
            signals = [s for s in signals if datetime.fromisoformat(s.get("timestamp", now.isoformat())) >= cutoff]
        
        # Count total signals
        total_signals = len(signals)
        if total_signals == 0:
            return {
                "total_signals": 0,
                "accuracy": 0,
                "win_rate": 0,
                "profit_factor": 0,
                "average_return": 0,
                "sharpe_ratio": 0
            }
        
        # Count successful signals
        successful = sum(1 for s in signals if s.get("outcome", {}).get("successful", False))
        
        # Calculate win rate
        win_rate = successful / total_signals if total_signals > 0 else 0
        
        # Calculate returns
        returns = [s.get("outcome", {}).get("return_pct", 0) for s in signals]
        average_return = sum(returns) / len(returns) if returns else 0
        
        # Calculate profit factor
        profits = sum(r for r in returns if r > 0)
        losses = abs(sum(r for r in returns if r < 0))
        profit_factor = profits / losses if losses > 0 else float('inf')
        
        # Calculate Sharpe ratio (assuming risk-free rate of 0%)
        if returns:
            returns_std = np.std(returns)
            sharpe_ratio = (average_return / returns_std) * np.sqrt(252) if returns_std > 0 else 0
        else:
            sharpe_ratio = 0
        
        # Compile metrics
        metrics = {
            "total_signals": total_signals,
            "accuracy": win_rate,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "average_return": average_return,
            "sharpe_ratio": sharpe_ratio
        }
        
        logger.info(f"Calculated performance metrics: {metrics}")
        return metrics
    
    def generate_performance_chart(self, signals, title="Signal Performance"):
        """
        Generate a performance chart for visualization.
        
        Args:
            signals (list): List of signal dictionaries with outcomes
            title (str): Chart title
            
        Returns:
            plotly.graph_objects.Figure: Plotly figure object
        """
        # Sort signals by timestamp
        signals = sorted(signals, key=lambda s: s.get("timestamp", ""))
        
        # Extract data for chart
        timestamps = [datetime.fromisoformat(s.get("timestamp", "")) for s in signals]
        returns = [s.get("outcome", {}).get("return_pct", 0) for s in signals]
        
        # Calculate cumulative returns
        cumulative_returns = [0]
        for r in returns:
            cumulative_returns.append(cumulative_returns[-1] + r)
        
        # Create chart
        fig = go.Figure()
        
        # Add cumulative returns line
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=cumulative_returns[1:],
            mode='lines',
            name='Cumulative Return',
            line=dict(color='blue', width=2)
        ))
        
        # Add individual signal markers
        buy_timestamps = [t for i, t in enumerate(timestamps) if signals[i].get("signal") == "BUY"]
        buy_returns = [returns[i] for i, t in enumerate(timestamps) if signals[i].get("signal") == "BUY"]
        
        sell_timestamps = [t for i, t in enumerate(timestamps) if signals[i].get("signal") == "SELL"]
        sell_returns = [returns[i] for i, t in enumerate(timestamps) if signals[i].get("signal") == "SELL"]
        
        fig.add_trace(go.Scatter(
            x=buy_timestamps,
            y=buy_returns,
            mode='markers',
            name='BUY Signals',
            marker=dict(color='green', size=8, symbol='triangle-up')
        ))
        
        fig.add_trace(go.Scatter(
            x=sell_timestamps,
            y=sell_returns,
            mode='markers',
            name='SELL Signals',
            marker=dict(color='red', size=8, symbol='triangle-down')
        ))
        
        # Update layout
        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title="Return (%)",
            legend_title="Legend",
            template="plotly_white"
        )
        
        return fig
    
    def get_signals_by_symbol(self, symbol, limit=100):
        """
        Get recent signals for a specific trading symbol.
        
        Args:
            symbol (str): Trading symbol (e.g., 'BTC/USDT')
            limit (int): Maximum number of signals to retrieve
            
        Returns:
            list: List of signal dictionaries
        """
        try:
            # Here you would query your database for signals
            # For now, we'll just return a mock response
            signals = [
                {
                    "id": "sig1",
                    "symbol": symbol,
                    "signal": "BUY",
                    "confidence": 0.85,
                    "price": 30000.00,
                    "timestamp": (datetime.now() - timedelta(days=7)).isoformat(),
                    "outcome": {
                        "successful": True,
                        "exit_price": 32000.00,
                        "exit_timestamp": datetime.now().isoformat(),
                        "return_pct": 6.67
                    }
                },
                {
                    "id": "sig2",
                    "symbol": symbol,
                    "signal": "SELL",
                    "confidence": 0.75,
                    "price": 32000.00,
                    "timestamp": (datetime.now() - timedelta(days=3)).isoformat(),
                    "outcome": {
                        "successful": False,
                        "exit_price": 33000.00,
                        "exit_timestamp": datetime.now().isoformat(),
                        "return_pct": -3.03
                    }
                }
            ]
            
            logger.info(f"Retrieved {len(signals)} signals for {symbol}")
            return signals
        except Exception as e:
            logger.error(f"Error retrieving signals for {symbol}: {str(e)}")
            return []
    
    def get_top_performing_signals(self, timeframe="month", limit=10):
        """
        Get the top performing signals within a specific timeframe.
        
        Args:
            timeframe (str): Time period to look at ('week', 'month', 'year')
            limit (int): Maximum number of signals to retrieve
            
        Returns:
            list: List of top performing signal dictionaries
        """
        try:
            # Here you would query your database for top signals
            # For now, we'll just return a mock response
            now = datetime.now()
            if timeframe == "week":
                cutoff = now - timedelta(days=7)
            elif timeframe == "month":
                cutoff = now - timedelta(days=30)
            else:  # year
                cutoff = now - timedelta(days=365)
            
            # Mock data
            signals = [
                {
                    "id": f"sig{i}",
                    "symbol": f"SYMBOL{i}",
                    "signal": "BUY" if i % 2 == 0 else "SELL",
                    "confidence": 0.7 + (i / 100),
                    "price": 1000 + (i * 100),
                    "timestamp": (now - timedelta(days=i)).isoformat(),
                    "outcome": {
                        "successful": True,
                        "exit_price": 1000 + (i * 100) * (1.1 if i % 2 == 0 else 0.9),
                        "exit_timestamp": now.isoformat(),
                        "return_pct": 10 - i
                    }
                }
                for i in range(1, limit + 1)
            ]
            
            # Sort by return percentage
            signals = sorted(signals, key=lambda s: s["outcome"]["return_pct"], reverse=True)
            
            logger.info(f"Retrieved {len(signals)} top performing signals")
            return signals
        except Exception as e:
            logger.error(f"Error retrieving top performing signals: {str(e)}")
            return []
    
    def export_performance_report(self, timeframe="month", format="csv"):
        """
        Export a performance report for a specific timeframe.
        
        Args:
            timeframe (str): Time period for the report ('week', 'month', 'year', 'all')
            format (str): Export format ('csv', 'json', 'html')
            
        Returns:
            str: Path to the exported file
        """
        try:
            # Get all signals for the timeframe
            # In a real implementation, this would query your database
            all_signals = []
            for i in range(1, 21):
                timestamp = datetime.now() - timedelta(days=i)
                all_signals.append({
                    "id": f"sig{i}",
                    "symbol": f"BTC/USDT" if i % 2 == 0 else "ETH/USDT",
                    "signal": "BUY" if i % 3 == 0 else "SELL",
                    "confidence": 0.7 + (i / 100),
                    "price": 30000 - (i * 100),
                    "timestamp": timestamp.isoformat(),
                    "outcome": {
                        "successful": i % 2 == 0,
                        "exit_price": 30000 - (i * 100) * (0.95 if i % 3 == 0 else 1.05),
                        "exit_timestamp": (timestamp + timedelta(days=1)).isoformat(),
                        "return_pct": 5 if i % 2 == 0 else -3
                    }
                })
            
            # Calculate metrics
            metrics = self.calculate_performance_metrics(all_signals, timeframe)
            
            # Create DataFrame
            df = pd.DataFrame(all_signals)
            
            # Add metrics to the DataFrame
            metrics_df = pd.DataFrame([metrics])
            
            # Export based on format
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            filename = f"performance_report_{timeframe}_{timestamp}.{format}"
            
            if format == "csv":
                df.to_csv(filename, index=False)
                metrics_df.to_csv(f"metrics_{filename}", index=False)
            elif format == "json":
                df.to_json(filename, orient="records")
                metrics_df.to_json(f"metrics_{filename}", orient="records")
            elif format == "html":
                html_content = f"""
                <html>
                <head>
                    <title>Performance Report - {timeframe}</title>
                    <style>
                        body {{ font-family: Arial, sans-serif; margin: 20px; }}
                        h1 {{ color: #333366; }}
                        table {{ border-collapse: collapse; width: 100%; }}
                        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                        tr:nth-child(even) {{ background-color: #f2f2f2; }}
                        th {{ background-color: #333366; color: white; }}
                    </style>
                </head>
                <body>
                    <h1>Performance Report - {timeframe.capitalize()}</h1>
                    <h2>Metrics Summary</h2>
                    <table>
                        <tr>
                            <th>Metric</th>
                            <th>Value</th>
                        </tr>
                        <tr>
                            <td>Total Signals</td>
                            <td>{metrics['total_signals']}</td>
                        </tr>
                        <tr>
                            <td>Win Rate</td>
                            <td>{metrics['win_rate']:.2%}</td>
                        </tr>
                        <tr>
                            <td>Profit Factor</td>
                            <td>{metrics['profit_factor']:.2f}</td>
                        </tr>
                        <tr>
                            <td>Average Return</td>
                            <td>{metrics['average_return']:.2%}</td>
                        </tr>
                        <tr>
                            <td>Sharpe Ratio</td>
                            <td>{metrics['sharpe_ratio']:.2f}</td>
                        </tr>
                    </table>
                    
                    <h2>Signal Details</h2>
                    <table>
                        <tr>
                            <th>ID</th>
                            <th>Symbol</th>
                            <th>Signal</th>
                            <th>Confidence</th>
                            <th>Price</th>
                            <th>Timestamp</th>
                            <th>Successful</th>
                            <th>Return %</th>
                        </tr>
                """
                
                for signal in all_signals:
                    outcome = signal.get("outcome", {})
                    html_content += f"""
                        <tr>
                            <td>{signal.get("id", "")}</td>
                            <td>{signal.get("symbol", "")}</td>
                            <td>{signal.get("signal", "")}</td>
                            <td>{signal.get("confidence", 0):.2%}</td>
                            <td>{signal.get("price", 0):.2f}</td>
                            <td>{signal.get("timestamp", "")}</td>
                            <td>{"Yes" if outcome.get("successful", False) else "No"}</td>
                            <td>{outcome.get("return_pct", 0):.2%}</td>
                        </tr>
                    """
                
                html_content += """
                    </table>
                </body>
                </html>
                """
                
                with open(filename, "w") as f:
                    f.write(html_content)
            
            logger.info(f"Exported performance report to {filename}")
            return filename
        except Exception as e:
            logger.error(f"Error exporting performance report: {str(e)}")
            raise


# Example usage
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize the performance tracker
    tracker = PerformanceTracker()
    
    # Example signal data
    signal_data = {
        "id": "sig123",
        "symbol": "BTC/USDT",
        "signal": "BUY",
        "confidence": 0.85,
        "price": 30000.00,
        "timestamp": datetime.now().isoformat()
    }
    
    # Record a signal
    tracker.record_signal(signal_data)
    
    # Example outcome data
    outcome_data = {
        "successful": True,
        "exit_price": 32000.00,
        "exit_timestamp": (datetime.now() + timedelta(days=1)).isoformat(),
        "return_pct": 6.67
    }
    
    # Record the outcome
    tracker.record_outcome(signal_data["id"], outcome_data)
    
    # Get signals for a symbol
    signals = tracker.get_signals_by_symbol("BTC/USDT")
    
    # Calculate performance metrics
    metrics = tracker.calculate_performance_metrics(signals)
    print(f"Performance metrics: {metrics}")
    
    # Export a performance report
    report_path = tracker.export_performance_report(timeframe="month", format="html")
    print(f"Performance report exported to: {report_path}")
