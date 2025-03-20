import random
import string
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
import logging
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import and_, or_, func

from app.models.redeem_code import RedeemCode, RedeemHistory, CodeStatus, CodeDuration
from app.models.subscription import Subscription, SubscriptionTier, SubscriptionStatus
from app.models.user import User

logger = logging.getLogger(__name__)

class RedeemCodeService:
    def __init__(self):
        """Initialize the redeem code service."""
        pass
    
    @staticmethod
    def generate_code(length: int = 12) -> str:
        """Generate a random alphanumeric code."""
        # Use uppercase letters and digits, exclude similar-looking characters
        chars = ''.join(set(string.ascii_uppercase + string.digits) - set('O0I1'))
        # Generate chunks of 4 characters separated by hyphens
        chunks = []
        for i in range(0, length, 4):
            chunk_len = min(4, length - i)
            chunks.append(''.join(random.choice(chars) for _ in range(chunk_len)))
        return '-'.join(chunks)
    
    async def create_codes(
        self, 
        db: AsyncSession, 
        tier: SubscriptionTier, 
        duration_type: CodeDuration,
        duration_days: Optional[int] = None,
        quantity: int = 1,
        max_uses: Optional[int] = 1,
        description: Optional[str] = None,
        valid_from: Optional[datetime] = None,
        valid_until: Optional[datetime] = None,
        created_by_id: str = None
    ) -> List[RedeemCode]:
        """
        Generate multiple redeem codes.
        
        Args:
            db: Database session
            tier: Subscription tier for this code
            duration_type: Type of duration (7_DAYS, 15_DAYS, 30_DAYS, FREE_FOREVER, CUSTOM)
            duration_days: Number of days this code provides (required if duration_type is CUSTOM)
            quantity: Number of codes to generate
            max_uses: Max number of times code can be redeemed (None = unlimited)
            description: Optional description for the codes
            valid_from: When code becomes valid (default: now)
            valid_until: When code expires (default: never)
            created_by_id: User ID who created these codes
            
        Returns:
            List of created redeem codes
        """
        if valid_from is None:
            valid_from = datetime.utcnow()
        
        # Determine actual duration in days
        if duration_type != CodeDuration.CUSTOM:
            days = CodeDuration.get_days(duration_type)
            if days is None:
                raise ValueError(f"Invalid duration type: {duration_type}")
            duration_days = days
        elif duration_days is None:
            raise ValueError("duration_days is required when duration_type is CUSTOM")
            
        codes = []
        for _ in range(quantity):
            # Generate a unique code
            while True:
                code_str = self.generate_code()
                # Check if code already exists in DB
                result = await db.execute(select(RedeemCode).filter(RedeemCode.code == code_str))
                if result.scalar_one_or_none() is None:
                    break
            
            # Create code object
            code = RedeemCode(
                code=code_str,
                subscription_tier=tier,
                duration_type=duration_type,
                duration_days=duration_days,
                max_uses=max_uses,
                description=description,
                valid_from=valid_from,
                valid_until=valid_until,
                created_by=created_by_id
            )
            
            db.add(code)
            codes.append(code)
        
        await db.commit()
        
        # Refresh to get generated IDs
        for code in codes:
            await db.refresh(code)
        
        return codes
    
    async def redeem_code(self, db: AsyncSession, code_str: str, user_id: str) -> Dict[str, Any]:
        """
        Redeem a code for a user.
        
        Args:
            db: Database session
            code_str: The code to redeem
            user_id: ID of the user redeeming the code
            
        Returns:
            Dict with result status and message
            
        Raises:
            ValueError: If code is invalid, expired, or already used
        """
        # Find the code
        result = await db.execute(select(RedeemCode).filter(RedeemCode.code == code_str))
        code = result.scalar_one_or_none()
        
        if not code:
            raise ValueError("Invalid redeem code")
        
        # Check if code is active
        if code.status != CodeStatus.ACTIVE:
            raise ValueError(f"This code is {code.status.value.lower()}")
        
        # Check if code is valid now
        now = datetime.utcnow()
        if code.valid_from and code.valid_from > now:
            raise ValueError("This code is not valid yet")
        
        if code.valid_until and code.valid_until < now:
            code.status = CodeStatus.EXPIRED
            await db.commit()
            raise ValueError("This code has expired")
        
        # Check if code has reached max uses
        if code.max_uses and code.uses_count >= code.max_uses:
            code.status = CodeStatus.USED
            await db.commit()
            raise ValueError("This code has reached its maximum number of uses")
        
        # Check if user has already used this code
        result = await db.execute(
            select(RedeemHistory).filter(
                RedeemHistory.code_id == code.id,
                RedeemHistory.user_id == user_id
            )
        )
        if result.scalar_one_or_none():
            raise ValueError("You have already redeemed this code")
        
        # Create or extend subscription
        result = await db.execute(
            select(Subscription).filter(
                Subscription.user_id == user_id,
                Subscription.status.in_([SubscriptionStatus.ACTIVE, SubscriptionStatus.TRIAL])
            ).order_by(Subscription.end_date.desc())
        )
        existing_sub = result.scalar_one_or_none()
        
        # Get user info
        result = await db.execute(select(User).filter(User.id == user_id))
        user = result.scalar_one_or_none()
        if not user:
            raise ValueError("User not found")
        
        subscription_info = {}
        
        if existing_sub and existing_sub.status in [SubscriptionStatus.ACTIVE, SubscriptionStatus.TRIAL]:
            # Extend existing subscription
            if existing_sub.tier != code.subscription_tier:
                subscription_info["tier_upgraded"] = True
                subscription_info["old_tier"] = existing_sub.tier
                subscription_info["new_tier"] = code.subscription_tier
                existing_sub.tier = code.subscription_tier
            
            if existing_sub.end_date < now:
                # Subscription expired but still in DB - start fresh from now
                existing_sub.start_date = now
                existing_sub.end_date = now + timedelta(days=code.duration_days)
            else:
                # Extend current subscription
                existing_sub.end_date = existing_sub.end_date + timedelta(days=code.duration_days)
            
            existing_sub.status = SubscriptionStatus.ACTIVE
            subscription_info["subscription_id"] = str(existing_sub.id)
            subscription_info["end_date"] = existing_sub.end_date
        else:
            # Create new subscription
            new_sub = Subscription(
                user_id=user_id,
                tier=code.subscription_tier,
                status=SubscriptionStatus.ACTIVE,
                payment_method="redeem_code",
                payment_id=code.code,
                start_date=now,
                end_date=now + timedelta(days=code.duration_days)
            )
            db.add(new_sub)
            subscription_info["subscription_id"] = "new"
            subscription_info["end_date"] = new_sub.end_date
            
        # Record redemption
        redemption = RedeemHistory(
            code_id=code.id,
            user_id=user_id
        )
        db.add(redemption)
        
        # Update code use count
        code.uses_count += 1
        if code.max_uses and code.uses_count >= code.max_uses:
            code.status = CodeStatus.USED
        
        await db.commit()
        
        # Return subscription details
        return {
            "status": "success",
            "message": f"Successfully redeemed code for {code.duration_days} days of {code.subscription_tier} subscription",
            "subscription": {
                "tier": code.subscription_tier,
                "duration_days": code.duration_days,
                "end_date": subscription_info["end_date"].strftime("%Y-%m-%d %H:%M:%S"),
                **subscription_info
            },
            "user": {
                "id": str(user.id),
                "username": user.username,
                "email": user.email
            },
            "code": {
                "id": str(code.id),
                "code": code.code,
                "uses_count": code.uses_count,
                "max_uses": code.max_uses
            }
        }
    
    async def get_codes(
        self, 
        db: AsyncSession, 
        skip: int = 0, 
        limit: int = 100, 
        status: Optional[CodeStatus] = None
    ) -> List[RedeemCode]:
        """Get redeem codes with optional filtering."""
        query = select(RedeemCode)
        
        if status:
            query = query.filter(RedeemCode.status == status)
        
        query = query.offset(skip).limit(limit)
        result = await db.execute(query)
        return result.scalars().all()

# Create the service instance
redeem_code_service = RedeemCodeService()
