from datetime import datetime
from enum import Enum
from typing import Optional
from pydantic import BaseModel, UUID4
from app.core.database import db

class SubscriptionTier(str, Enum):
    FREE = "FREE"
    BASIC = "BASIC"
    PREMIUM = "PREMIUM"
    VIP = "VIP"

class SubscriptionStatus(str, Enum):
    ACTIVE = "ACTIVE"
    EXPIRED = "EXPIRED"
    CANCELED = "CANCELED"
    TRIAL = "TRIAL"

class SubscriptionBase(BaseModel):
    """Base Subscription Model"""
    user_id: UUID4
    tier: SubscriptionTier = SubscriptionTier.FREE
    status: SubscriptionStatus = SubscriptionStatus.TRIAL
    payment_method: Optional[str] = None
    payment_id: Optional[str] = None
    amount: Optional[float] = None
    currency: str = "USD"
    start_date: datetime
    end_date: datetime

class SubscriptionCreate(SubscriptionBase):
    """Subscription Creation Model"""
    pass

class SubscriptionDB(SubscriptionBase):
    """Subscription Database Model"""
    id: UUID4
    created_at: datetime
    updated_at: datetime

    class Config:
        orm_mode = True

class SubscriptionResponse(SubscriptionBase):
    """Subscription Response Model"""
    id: UUID4
    created_at: datetime
    updated_at: datetime

    class Config:
        orm_mode = True

class SubscriptionService:
    """Subscription Database Operations"""
    TABLE = "subscriptions"

    @staticmethod
    async def create(subscription_data: dict) -> dict:
        """Create a new subscription"""
        subscription_data['created_at'] = datetime.utcnow()
        subscription_data['updated_at'] = datetime.utcnow()
        return await db.execute(SubscriptionService.TABLE, 'insert', subscription_data)

    @staticmethod
    async def get_by_id(subscription_id: str) -> Optional[dict]:
        """Get subscription by ID"""
        result = await db.execute(SubscriptionService.TABLE, 'select',
                                filters={'id': subscription_id})
        return result.data[0] if result and result.data else None

    @staticmethod
    async def get_by_user_id(user_id: str) -> list:
        """Get all subscriptions for a user"""
        result = await db.execute(SubscriptionService.TABLE, 'select',
                                filters={'user_id': user_id})
        return result.data if result else []

    @staticmethod
    async def update(subscription_id: str, update_data: dict) -> dict:
        """Update subscription data"""
        update_data['updated_at'] = datetime.utcnow()
        return await db.execute(SubscriptionService.TABLE, 'update',
                              data=update_data,
                              filters={'id': subscription_id})

    @staticmethod
    async def delete(subscription_id: str) -> dict:
        """Delete subscription"""
        return await db.execute(SubscriptionService.TABLE, 'delete',
                              filters={'id': subscription_id})

    @staticmethod
    async def list_active_subscriptions() -> list:
        """List all active subscriptions"""
        result = await db.execute(SubscriptionService.TABLE, 'select',
                                filters={'status': SubscriptionStatus.ACTIVE})
        return result.data if result else []

    @staticmethod
    async def expire_subscriptions() -> list:
        """Find and expire subscriptions that have passed their end_date"""
        now = datetime.utcnow()
        result = await db.execute(
            SubscriptionService.TABLE,
            'select',
            filters={
                'status': SubscriptionStatus.ACTIVE,
                'end_date.lt': now.isoformat()
            }
        )
        
        if result and result.data:
            expired_ids = [sub['id'] for sub in result.data]
            update_data = {
                'status': SubscriptionStatus.EXPIRED,
                'updated_at': now
            }
            for sub_id in expired_ids:
                await db.execute(
                    SubscriptionService.TABLE,
                    'update',
                    data=update_data,
                    filters={'id': sub_id}
                )
        
        return result.data if result else []
