"""
Dashboard module for the AI Signal Provider.
This Streamlit dashboard visualizes trading signals and performance metrics.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Add the parent directory to the path to import from other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project modules
try:
    from monitoring.performance_tracker import PerformanceTracker
    from database.db_manager import DBManager
    from signal_generation.signal_generator import SignalGenerator
except ImportError as e:
    st.error(f"Error importing modules: {e}")
    logging.error(f"Error importing modules: {e}")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="AI Signal Provider Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize database manager and performance tracker
@st.cache_resource
def init_resources():
    try:
        db_manager = DBManager()
        performance_tracker = PerformanceTracker(db_manager)
        return db_manager, performance_tracker
    except Exception as e:
        st.error(f"Error initializing resources: {e}")
        logger.error(f"Error initializing resources: {e}")
        return None, None

# Custom styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #333366;
        margin-bottom: 1rem;
    }
    .section-header {
        font-size: 1.8rem;
        font-weight: bold;
        color: #333366;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-box {
        background-color: #f0f2f6;
        border-radius: 5px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #333366;
    }
    .metric-label {
        font-size: 1rem;
        color: #666;
    }
    .positive {
        color: #28a745;
    }
    .negative {
        color: #dc3545;
    }
    .warning {
        color: #ffc107;
    }
    .info-box {
        background-color: #e7f0fd;
        border-left: 5px solid #4285f4;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 5px 5px 0;
    }
</style>
""", unsafe_allow_html=True)

def main():
    """Main function for the dashboard."""
    # Initialize resources
    db_manager, performance_tracker = init_resources()
    
    # Sidebar
    with st.sidebar:
        st.image("https://via.placeholder.com/150x150.png?text=AI+Signals", width=150)
        st.markdown("## AI Signal Provider")
        
        # Timeframe selection
        st.markdown("### Timeframe")
        timeframe = st.selectbox(
            "Select timeframe",
            options=["day", "week", "month", "year", "all"],
            index=2  # Default to month
        )
        
        # Symbol selection
        st.markdown("### Trading Pair")
        symbol = st.selectbox(
            "Select trading pair",
            options=["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "ADA/USDT", "DOGE/USDT"],
            index=0  # Default to BTC/USDT
        )
        
        # Date range selection
        st.markdown("### Date Range")
        today = datetime.now().date()
        start_date = st.date_input(
            "Start date",
            value=today - timedelta(days=30),
            min_value=today - timedelta(days=365),
            max_value=today
        )
        end_date = st.date_input(
            "End date",
            value=today,
            min_value=start_date,
            max_value=today
        )
        
        # Refresh button
        if st.button("Refresh Data"):
            st.cache_resource.clear()
            st.experimental_rerun()
    
    # Header
    st.markdown('<div class="main-header">AI Signal Provider Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="info-box">Real-time monitoring and analysis of AI-generated trading signals.</div>', unsafe_allow_html=True)
    
    # Overview section with metrics
    st.markdown('<div class="section-header">Performance Overview</div>', unsafe_allow_html=True)
    
    # Example metrics - in a real implementation, these would come from the database
    # Creating a mock metrics dictionary for demonstration
    metrics = {
        "total_signals": 128,
        "accuracy": 0.78,
        "win_rate": 0.75,
        "profit_factor": 2.3,
        "average_return": 0.042,
        "sharpe_ratio": 1.8
    }
    
    # Display metrics in a row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f'''
        <div class="metric-box">
            <div class="metric-value">{metrics["total_signals"]}</div>
            <div class="metric-label">Total Signals</div>
        </div>
        ''', unsafe_allow_html=True)
    
    with col2:
        win_rate_color = "positive" if metrics["win_rate"] >= 0.7 else "warning" if metrics["win_rate"] >= 0.5 else "negative"
        st.markdown(f'''
        <div class="metric-box">
            <div class="metric-value {win_rate_color}">{metrics["win_rate"]:.1%}</div>
            <div class="metric-label">Win Rate</div>
        </div>
        ''', unsafe_allow_html=True)
    
    with col3:
        avg_return_color = "positive" if metrics["average_return"] > 0 else "negative"
        st.markdown(f'''
        <div class="metric-box">
            <div class="metric-value {avg_return_color}">{metrics["average_return"]:.2%}</div>
            <div class="metric-label">Avg. Return</div>
        </div>
        ''', unsafe_allow_html=True)
    
    with col4:
        sharpe_color = "positive" if metrics["sharpe_ratio"] > 1 else "warning" if metrics["sharpe_ratio"] > 0 else "negative"
        st.markdown(f'''
        <div class="metric-box">
            <div class="metric-value {sharpe_color}">{metrics["sharpe_ratio"]:.2f}</div>
            <div class="metric-label">Sharpe Ratio</div>
        </div>
        ''', unsafe_allow_html=True)
    
    # Performance chart
    st.markdown('<div class="section-header">Signal Performance</div>', unsafe_allow_html=True)
    
    # Create mock data for a performance chart
    num_days = (end_date - start_date).days + 1
    dates = [start_date + timedelta(days=i) for i in range(num_days)]
    
    # Generate mock cumulative returns
    np.random.seed(42)  # For reproducibility
    returns = [0]
    for i in range(1, len(dates)):
        # Add some randomness but with an upward bias
        daily_return = np.random.normal(0.002, 0.01)
        returns.append(returns[-1] + daily_return)
    
    # Create a Plotly figure
    fig = go.Figure()
    
    # Add cumulative returns line
    fig.add_trace(go.Scatter(
        x=dates,
        y=returns,
        mode='lines',
        name='Cumulative Return',
        line=dict(color='blue', width=2)
    ))
    
    # Add mock buy signals
    buy_dates = [dates[i] for i in range(0, len(dates), 5)]  # Every 5 days
    buy_returns = [returns[i] for i in range(0, len(dates), 5)]
    
    fig.add_trace(go.Scatter(
        x=buy_dates,
        y=buy_returns,
        mode='markers',
        name='BUY Signals',
        marker=dict(color='green', size=8, symbol='triangle-up')
    ))
    
    # Add mock sell signals
    sell_dates = [dates[i] for i in range(2, len(dates), 5)]  # Every 5 days with offset
    sell_returns = [returns[i] for i in range(2, len(dates), 5)]
    
    fig.add_trace(go.Scatter(
        x=sell_dates,
        y=sell_returns,
        mode='markers',
        name='SELL Signals',
        marker=dict(color='red', size=8, symbol='triangle-down')
    ))
    
    # Update layout
    fig.update_layout(
        title=f"Performance for {symbol} ({start_date} to {end_date})",
        xaxis_title="Date",
        yaxis_title="Cumulative Return",
        legend_title="Legend",
        template="plotly_white",
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Recent signals
    st.markdown('<div class="section-header">Recent Signals</div>', unsafe_allow_html=True)
    
    # Create mock data for recent signals
    mock_signals = []
    for i in range(10):
        days_ago = 10 - i
        timestamp = datetime.now() - timedelta(days=days_ago)
        signal_type = "BUY" if i % 3 != 0 else "SELL"
        price = 30000 + (i * 100) if signal_type == "BUY" else 31000 - (i * 50)
        successful = i % 4 != 0  # 75% success rate
        return_pct = 2.5 if successful and signal_type == "BUY" else -1.5 if not successful and signal_type == "BUY" else -2.0 if successful and signal_type == "SELL" else 1.0
        
        mock_signals.append({
            "id": f"sig{i}",
            "symbol": symbol,
            "signal_type": signal_type,
            "price": price,
            "timestamp": timestamp,
            "confidence": 0.7 + (i / 50),
            "successful": successful,
            "return_pct": return_pct
        })
    
    # Convert to DataFrame
    df_signals = pd.DataFrame(mock_signals)
    
    # Format the dataframe
    df_display = df_signals.copy()
    df_display["timestamp"] = df_display["timestamp"].dt.strftime("%Y-%m-%d %H:%M")
    df_display["confidence"] = df_display["confidence"].apply(lambda x: f"{x:.2%}")
    df_display["price"] = df_display["price"].apply(lambda x: f"${x:.2f}")
    df_display["return_pct"] = df_display["return_pct"].apply(lambda x: f"{x:.2%}")
    df_display["successful"] = df_display["successful"].apply(lambda x: "✅" if x else "❌")
    
    # Rename columns for display
    df_display = df_display.rename(columns={
        "id": "ID",
        "symbol": "Symbol",
        "signal_type": "Signal",
        "price": "Price",
        "timestamp": "Date & Time",
        "confidence": "Confidence",
        "successful": "Success",
        "return_pct": "Return"
    })
    
    # Display the table
    st.dataframe(df_display, use_container_width=True)
    
    # Signal distribution section
    st.markdown('<div class="section-header">Signal Analysis</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Create a pie chart for signal distribution
        signal_counts = df_signals["signal_type"].value_counts().reset_index()
        signal_counts.columns = ["Signal", "Count"]
        
        fig_pie = px.pie(
            signal_counts,
            values="Count",
            names="Signal",
            title="Signal Distribution",
            color="Signal",
            color_discrete_map={"BUY": "green", "SELL": "red"},
            hole=0.4
        )
        
        fig_pie.update_layout(
            legend_title="Signal Type",
            template="plotly_white"
        )
        
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # Create a bar chart for success rate by signal type
        success_by_type = df_signals.groupby("signal_type")["successful"].mean().reset_index()
        success_by_type.columns = ["Signal", "Success Rate"]
        success_by_type["Success Rate"] = success_by_type["Success Rate"] * 100
        
        fig_bar = px.bar(
            success_by_type,
            x="Signal",
            y="Success Rate",
            title="Success Rate by Signal Type",
            color="Signal",
            color_discrete_map={"BUY": "green", "SELL": "red"},
            text_auto='.1f'
        )
        
        fig_bar.update_layout(
            xaxis_title="Signal Type",
            yaxis_title="Success Rate (%)",
            yaxis=dict(range=[0, 100]),
            template="plotly_white"
        )
        
        st.plotly_chart(fig_bar, use_container_width=True)
    
    # Monthly performance section
    st.markdown('<div class="section-header">Monthly Performance</div>', unsafe_allow_html=True)
    
    # Create mock data for monthly performance
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    current_month = datetime.now().month
    
    # Only show data for past months up to the current month
    display_months = months[:current_month]
    monthly_returns = np.random.normal(0.03, 0.02, len(display_months))
    
    # Create dataframe
    df_monthly = pd.DataFrame({
        "Month": display_months,
        "Return": monthly_returns
    })
    
    # Create a bar chart for monthly returns
    fig_monthly = px.bar(
        df_monthly,
        x="Month",
        y="Return",
        title="Monthly Returns",
        color="Return",
        color_continuous_scale=["red", "lightgrey", "green"],
        text_auto='.2%'
    )
    
    fig_monthly.update_layout(
        xaxis_title="Month",
        yaxis_title="Return (%)",
        template="plotly_white"
    )
    
    st.plotly_chart(fig_monthly, use_container_width=True)
    
    # Export options
    st.markdown('<div class="section-header">Export Options</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Export Signals as CSV"):
            st.info("This would export the signals data as a CSV file.")
    
    with col2:
        if st.button("Export Performance Report"):
            st.info("This would generate and export a comprehensive performance report.")
    
    with col3:
        if st.button("Export Chart Image"):
            st.info("This would export the performance chart as an image file.")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "© 2023 AI Signal Provider | Dashboard refreshed: " + 
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )

if __name__ == "__main__":
    main()
