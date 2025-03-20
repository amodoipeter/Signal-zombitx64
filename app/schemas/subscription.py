from pydantic import BaseModel
from typing import Optional
import uuid
from datetime import datetime

from app.models.subscription import SubscriptionTier, SubscriptionStatus

class SubscriptionBase(BaseModel):
    tier: SubscriptionTier
    status: SubscriptionStatus
    payment_method: Optional[str] = None
    payment_id: Optional[str] = None
    amount: Optional[float] = None
    currency: str = "USD"
    start_date: datetime
    end_date: datetime

class SubscriptionCreate(BaseModel):
    tier: SubscriptionTier
    payment_method: Optional[str] = None
    payment_id: Optional[str] = None
    amount: Optional[float] = None
    currency: str = "USD"
    duration_months: int = 1  # Default to 1 month subscription

class SubscriptionUpdate(BaseModel):
    status: Optional[SubscriptionStatus] = None
    payment_id: Optional[str] = None
    end_date: Optional[datetime] = None

class SubscriptionInDBBase(SubscriptionBase):
    id: uuid.UUID
    user_id: uuid.UUID
    created_at: datetime
    updated_at: datetime
    
    class Config:
        orm_mode = True

class Subscription(SubscriptionInDBBase):
    pass

class SubscriptionWithUser(SubscriptionInDBBase):
    user: "User"  # This will be filled by the ORM
