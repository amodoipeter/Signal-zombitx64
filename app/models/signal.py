from datetime import datetime
from enum import Enum
from typing import Optional, Dict
from pydantic import BaseModel, UUID4, Field

from app.core.database import db

class SignalType(str, Enum):
    BUY = "BUY"
    SELL = "SELL"

class SignalStatus(str, Enum):
    ACTIVE = "ACTIVE"
    TP_HIT = "TP_HIT"
    SL_HIT = "SL_HIT"
    EXPIRED = "EXPIRED"
    CANCELED = "CANCELED"

class SignalBase(BaseModel):
    """Base Signal Model"""
    symbol: str
    market: str
    signal_type: SignalType
    entry_price: float
    take_profit: float
    stop_loss: float
    risk_reward_ratio: float
    timeframe: str
    status: SignalStatus = SignalStatus.ACTIVE
    confidence_score: int = Field(ge=0, le=100)
    analysis_summary: Optional[str] = None
    chart_url: Optional[str] = None
    entry_time: datetime = Field(default_factory=datetime.utcnow)
    close_time: Optional[datetime] = None
    profit_loss: Optional[float] = None
    indicators_data: Optional[Dict] = None
    ai_model_version: str
    strategy_name: str
    strategy_category: str

class SignalCreate(SignalBase):
    """Signal Creation Model"""
    pass

class SignalDB(SignalBase):
    """Signal Database Model"""
    id: UUID4
    created_at: datetime
    updated_at: datetime

    class Config:
        orm_mode = True

class SignalResponse(SignalBase):
    """Signal Response Model"""
    id: UUID4
    created_at: datetime
    updated_at: datetime

    class Config:
        orm_mode = True

class SignalService:
    """Signal Database Operations"""
    TABLE = "signals"

    @staticmethod
    async def create(signal_data: dict) -> dict:
        """Create a new signal"""
        signal_data['created_at'] = datetime.utcnow()
        signal_data['updated_at'] = datetime.utcnow()
        return await db.execute(SignalService.TABLE, 'insert', signal_data)

    @staticmethod
    async def get_by_id(signal_id: str) -> Optional[dict]:
        """Get signal by ID"""
        result = await db.execute(SignalService.TABLE, 'select',
                                filters={'id': signal_id})
        return result.data[0] if result and result.data else None

    @staticmethod
    async def get_active_signals() -> list:
        """Get all active signals"""
        result = await db.execute(SignalService.TABLE, 'select',
                                filters={'status': SignalStatus.ACTIVE})
        return result.data if result else []

    @staticmethod
    async def update(signal_id: str, update_data: dict) -> dict:
        """Update signal data"""
        update_data['updated_at'] = datetime.utcnow()
        return await db.execute(SignalService.TABLE, 'update',
                              data=update_data,
                              filters={'id': signal_id})

    @staticmethod
    async def delete(signal_id: str) -> dict:
        """Delete signal"""
        return await db.execute(SignalService.TABLE, 'delete',
                              filters={'id': signal_id})

    @staticmethod
    async def list_signals(filters: dict = None) -> list:
        """List signals with optional filters"""
        result = await db.execute(SignalService.TABLE, 'select', filters=filters)
        return result.data if result else []

    @staticmethod
    async def get_signals_by_symbol(symbol: str) -> list:
        """Get signals for a specific symbol"""
        result = await db.execute(SignalService.TABLE, 'select',
                                filters={'symbol': symbol})
        return result.data if result else []

    @staticmethod
    async def update_signal_status(signal_id: str, status: SignalStatus,
                                 profit_loss: float = None) -> dict:
        """Update signal status and profit/loss"""
        update_data = {
            'status': status,
            'updated_at': datetime.utcnow()
        }
        if status in [SignalStatus.TP_HIT, SignalStatus.SL_HIT]:
            update_data.update({
                'close_time': datetime.utcnow(),
                'profit_loss': profit_loss
            })
        return await db.execute(SignalService.TABLE, 'update',
                              data=update_data,
                              filters={'id': signal_id})

    @staticmethod
    async def cleanup_expired_signals(expiry_hours: int = 24) -> list:
        """Clean up old signals that have expired"""
        expiry_time = datetime.utcnow() - timedelta(hours=expiry_hours)
        result = await db.execute(
            SignalService.TABLE,
            'select',
            filters={
                'status': SignalStatus.ACTIVE,
                'entry_time.lt': expiry_time.isoformat()
            }
        )
        
        if result and result.data:
            expired_ids = [signal['id'] for signal in result.data]
            update_data = {
                'status': SignalStatus.EXPIRED,
                'updated_at': datetime.utcnow()
            }
            for signal_id in expired_ids:
                await db.execute(
                    SignalService.TABLE,
                    'update',
                    data=update_data,
                    filters={'id': signal_id}
                )
        
        return result.data if result else []
