from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
import uuid
from datetime import datetime

from app.models.signal import SignalType, SignalStatus

class SignalBase(BaseModel):
    symbol: str
    market: str
    signal_type: SignalType
    entry_price: float
    take_profit: float
    stop_loss: float
    risk_reward_ratio: float
    timeframe: str
    confidence_score: int = Field(..., ge=1, le=100)
    analysis_summary: str
    chart_url: Optional[str] = None
    ai_model_version: str
    strategy_name: str
    strategy_category: str

class SignalCreate(SignalBase):
    pass

class SignalUpdate(BaseModel):
    status: Optional[SignalStatus] = None
    close_time: Optional[datetime] = None
    profit_loss: Optional[float] = None
    chart_url: Optional[str] = None

class SignalInDBBase(SignalBase):
    id: uuid.UUID
    status: SignalStatus
    entry_time: datetime
    created_at: datetime
    updated_at: datetime
    indicators_data: Optional[Dict[str, Any]] = None
    close_time: Optional[datetime] = None
    profit_loss: Optional[float] = None
    
    class Config:
        orm_mode = True

class Signal(SignalInDBBase):
    pass

class SignalResponse(SignalInDBBase):
    pass
