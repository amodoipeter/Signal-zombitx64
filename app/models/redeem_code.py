import uuid
from datetime import datetime
from enum import Enum as PyEnum
from sqlalchemy import Column, String, Boolean, DateTime, Enum, Integer, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from app.core.database import Base
from app.models.subscription import SubscriptionTier

class CodeStatus(str, PyEnum):
    ACTIVE = "ACTIVE"
    USED = "USED"
    EXPIRED = "EXPIRED"
    DISABLED = "DISABLED"

class CodeDuration(str, PyEnum):
    SEVEN_DAYS = "SEVEN_DAYS"
    FIFTEEN_DAYS = "FIFTEEN_DAYS"
    THIRTY_DAYS = "THIRTY_DAYS"
    FREE_FOREVER = "FREE_FOREVER"
    CUSTOM = "CUSTOM"
    
    @classmethod
    def get_days(cls, duration_type):
        """Convert duration type to number of days"""
        duration_days = {
            cls.SEVEN_DAYS: 7,
            cls.FIFTEEN_DAYS: 15,
            cls.THIRTY_DAYS: 30,
            cls.FREE_FOREVER: 36500,  # ~100 years (essentially forever)
            cls.CUSTOM: None  # Custom duration requires explicit days
        }
        return duration_days.get(duration_type)

class RedeemCode(Base):
    __tablename__ = "redeem_codes"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    code = Column(String, unique=True, index=True)  # The actual redeem code
    subscription_tier = Column(Enum(SubscriptionTier))
    duration_type = Column(Enum(CodeDuration), default=CodeDuration.CUSTOM)
    duration_days = Column(Integer)  # How many days this code provides
    
    # Allow limiting codes to specific number of uses
    max_uses = Column(Integer, nullable=True)  # NULL means unlimited
    uses_count = Column(Integer, default=0)
    
    status = Column(Enum(CodeStatus), default=CodeStatus.ACTIVE)
    description = Column(String, nullable=True)  # Optional description/purpose of code
    
    # Validity period
    valid_from = Column(DateTime, default=datetime.utcnow)
    valid_until = Column(DateTime, nullable=True)  # NULL means no expiry
    
    # Audit fields
    created_by = Column(UUID(as_uuid=True), ForeignKey("users.id"))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Redeem history
    redemptions = relationship("RedeemHistory", back_populates="redeem_code")
    
    def __repr__(self):
        return f"<RedeemCode {self.code} - Tier: {self.subscription_tier} Status: {self.status}>"

class RedeemHistory(Base):
    __tablename__ = "redeem_history"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    code_id = Column(UUID(as_uuid=True), ForeignKey("redeem_codes.id"))
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"))
    redeemed_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    redeem_code = relationship("RedeemCode", back_populates="redemptions")
    user = relationship("User")
    
    def __repr__(self):
        return f"<RedeemHistory {self.id} - User: {self.user_id} Code: {self.code_id}>"
