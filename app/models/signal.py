import uuid
from datetime import datetime
from enum import Enum as PyEnum
from sqlalchemy import Column, String, Float, DateTime, Enum, Text, Boolean, Integer, ForeignKey
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship

from app.core.database import Base

class SignalType(str, PyEnum):
    BUY = "BUY"
    SELL = "SELL"

class SignalStatus(str, PyEnum):
    ACTIVE = "ACTIVE"
    TP_HIT = "TP_HIT"
    SL_HIT = "SL_HIT"
    EXPIRED = "EXPIRED"
    CANCELED = "CANCELED"

class Signal(Base):
    __tablename__ = "signals"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    symbol = Column(String, index=True)  # e.g., "BTCUSDT"
    market = Column(String, index=True)  # e.g., "crypto", "forex", "stocks"
    signal_type = Column(Enum(SignalType), index=True)
    entry_price = Column(Float)
    take_profit = Column(Float)
    stop_loss = Column(Float)
    risk_reward_ratio = Column(Float)
    timeframe = Column(String)  # e.g., "1h", "4h", "1d"
    status = Column(Enum(SignalStatus), default=SignalStatus.ACTIVE, index=True)
    confidence_score = Column(Integer)  # 1-100 score from AI
    analysis_summary = Column(Text)
    chart_url = Column(String, nullable=True)  # URL to chart image
    
    # Signal tracking
    entry_time = Column(DateTime, default=datetime.utcnow)
    close_time = Column(DateTime, nullable=True)
    profit_loss = Column(Float, nullable=True)  # Final P/L result in percentage
    
    # AI Data
    indicators_data = Column(JSONB, nullable=True)  # Store indicators values
    ai_model_version = Column(String)  # Version of AI model that generated the signal
    
    # Strategy info
    strategy_name = Column(String, index=True)  # e.g., "Ichimoku Cloud Breakout"
    strategy_category = Column(String, index=True)  # e.g., "Scalping", "Swing", "Trend"
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f"<Signal {self.symbol} {self.signal_type} at {self.entry_price}>"
