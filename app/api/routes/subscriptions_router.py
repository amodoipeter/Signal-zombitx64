from fastapi import APIRouter, Depends, HTTPException, status, Body, Request
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from datetime import datetime, timedelta
import uuid
from typing import List, Optional

from app.api.deps import get_current_active_user, get_current_superuser
from app.core.database import get_db
from app.models.subscription import Subscription, SubscriptionTier, SubscriptionStatus
from app.models.user import User
from app.schemas.subscription import (
    SubscriptionCreate, 
    Subscription as SubscriptionSchema, 
    SubscriptionUpdate
)
from app.services.payment.stripe_service import stripe_service

router = APIRouter()

@router.get("/", response_model=List[SubscriptionSchema])
async def get_user_subscriptions(
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """Get all subscriptions for the current user."""
    result = await db.execute(select(Subscription).filter(Subscription.user_id == current_user.id))
    subscriptions = result.scalars().all()
    return subscriptions

@router.post("/", response_model=SubscriptionSchema)
async def create_subscription(
    subscription_in: SubscriptionCreate,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """Create a new subscription."""
    # Check if user already has an active subscription
    result = await db.execute(
        select(Subscription).filter(
            Subscription.user_id == current_user.id,
            Subscription.status == SubscriptionStatus.ACTIVE,
            Subscription.end_date > datetime.utcnow()
        )
    )
    existing_subscription = result.scalar_one_or_none()
    
    if existing_subscription and existing_subscription.tier == subscription_in.tier:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User already has an active subscription for this tier"
        )
    
    # Calculate subscription dates
    start_date = datetime.utcnow()
    end_date = start_date + timedelta(days=30 * subscription_in.duration_months)
    
    # Create subscription
    subscription = Subscription(
        user_id=current_user.id,
        tier=subscription_in.tier,
        status=SubscriptionStatus.ACTIVE,
        payment_method=subscription_in.payment_method,
        payment_id=subscription_in.payment_id,
        amount=subscription_in.amount,
        currency=subscription_in.currency,
        start_date=start_date,
        end_date=end_date,
    )
    
    db.add(subscription)
    await db.commit()
    await db.refresh(subscription)
    
    return subscription

@router.get("/checkout/{tier}")
async def create_checkout_session(
    tier: SubscriptionTier,
    current_user: User = Depends(get_current_active_user),
):
    """Create a Stripe checkout session for subscription."""
    # Determine price based on tier
    prices = {
        SubscriptionTier.FREE: 0,
        SubscriptionTier.BASIC: 2999,  # $29.99
        SubscriptionTier.PREMIUM: 4999,  # $49.99
        SubscriptionTier.VIP: 9999,  # $99.99
    }
    
    if tier == SubscriptionTier.FREE:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot create checkout for free tier"
        )
    
    # Create checkout session
    checkout_session = await stripe_service.create_checkout_session(
        customer_email=current_user.email,
        price_amount=prices[tier],
        product_name=f"ZombitX64 {tier} Subscription",
        metadata={
            "user_id": str(current_user.id),
            "subscription_tier": tier,
        }
    )
    
    return {"checkout_url": checkout_session.url}

@router.post("/webhook", status_code=status.HTTP_200_OK)
async def stripe_webhook(
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """Handle Stripe webhook events."""
    # Get payload
    payload = await request.body()
    sig_header = request.headers.get("stripe-signature")
    
    # Process webhook
    event = await stripe_service.construct_event(payload, sig_header)
    
    # Handle checkout.session.completed
    if event["type"] == "checkout.session.completed":
        session = event["data"]["object"]
        
        # Get user ID and subscription tier from metadata
        user_id = session["metadata"].get("user_id")
        subscription_tier = session["metadata"].get("subscription_tier")
        
        if not user_id or not subscription_tier:
            return {"status": "error", "message": "Missing metadata"}
        
        # Create subscription
        try:
            user_uuid = uuid.UUID(user_id)
            
            # Calculate subscription dates
            start_date = datetime.utcnow()
            end_date = start_date + timedelta(days=30)  # 1 month subscription
            
            subscription = Subscription(
                user_id=user_uuid,
                tier=subscription_tier,
                status=SubscriptionStatus.ACTIVE,
                payment_method="stripe",
                payment_id=session["id"],
                amount=session["amount_total"] / 100,  # Convert cents to dollars
                currency=session["currency"].upper(),
                start_date=start_date,
                end_date=end_date,
            )
            
            db.add(subscription)
            await db.commit()
            
            return {"status": "success"}
            
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    return {"status": "success"}

@router.put("/{subscription_id}", response_model=SubscriptionSchema)
async def update_subscription(
    subscription_id: uuid.UUID,
    subscription_update: SubscriptionUpdate,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """Update a subscription."""
    # First check if subscription belongs to the current user
    result = await db.execute(
        select(Subscription).filter(
            Subscription.id == subscription_id,
            Subscription.user_id == current_user.id
        )
    )
    subscription = result.scalar_one_or_none()
    
    if not subscription:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Subscription not found"
        )
    
    # Update subscription fields
    for field, value in subscription_update.dict(exclude_unset=True).items():
        setattr(subscription, field, value)
    
    await db.commit()
    await db.refresh(subscription)
    
    return subscription

@router.delete("/{subscription_id}", status_code=status.HTTP_204_NO_CONTENT)
async def cancel_subscription(
    subscription_id: uuid.UUID,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """Cancel a subscription."""
    # First check if subscription belongs to the current user
    result = await db.execute(
        select(Subscription).filter(
            Subscription.id == subscription_id,
            Subscription.user_id == current_user.id
        )
    )
    subscription = result.scalar_one_or_none()
    
    if not subscription:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Subscription not found"
        )
    
    # Set subscription status to canceled
    subscription.status = SubscriptionStatus.CANCELED
    
    await db.commit()
    
    return None
