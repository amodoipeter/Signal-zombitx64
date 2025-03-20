from pydantic import BaseModel, validator, Field
from typing import Optional, List, Literal
import uuid
from datetime import datetime

from app.models.redeem_code import CodeStatus, CodeDuration
from app.models.subscription import SubscriptionTier

class RedeemCodeBase(BaseModel):
    subscription_tier: SubscriptionTier
    duration_type: CodeDuration
    max_uses: Optional[int] = Field(None, ge=1, description="Maximum number of times this code can be used (null = unlimited)")
    description: Optional[str] = None
    valid_from: Optional[datetime] = None
    valid_until: Optional[datetime] = None

class RedeemCodeCreate(RedeemCodeBase):
    quantity: int = Field(1, ge=1, le=100, description="Number of codes to generate")

class RedeemCodeUpdate(BaseModel):
    status: Optional[CodeStatus] = None
    description: Optional[str] = None
    valid_until: Optional[datetime] = None
    max_uses: Optional[int] = Field(None, ge=1)

class RedeemCodeInDBBase(RedeemCodeBase):
    id: uuid.UUID
    code: str
    status: CodeStatus
    uses_count: int
    created_by: uuid.UUID
    created_at: datetime
    updated_at: datetime
    
    class Config:
        orm_mode = True

class RedeemCode(RedeemCodeInDBBase):
    pass

class RedeemCodePublic(BaseModel):
    """Limited version of redeem code info for public usage"""
    code: str
    subscription_tier: SubscriptionTier
    duration_days: int
    valid_until: Optional[datetime] = None
    
    class Config:
        orm_mode = True

class RedeemRequest(BaseModel):
    code: str = Field(..., min_length=6, max_length=32)
