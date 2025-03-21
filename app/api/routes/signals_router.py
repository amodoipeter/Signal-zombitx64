from fastapi import APIRouter, Depends, HTTPException, status, Query, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import and_, desc
from typing import List, Optional
import uuid
from datetime import datetime
import logging

# Setup logger
logger = logging.getLogger(__name__)

from app.api.deps import get_current_active_user, get_current_superuser
from app.core.database import get_db
from app.models.signal import Signal, SignalStatus, SignalType
from app.models.user import User
from app.models.subscription import Subscription, SubscriptionStatus, SubscriptionTier
from app.schemas.signal import SignalCreate, SignalResponse, SignalUpdate
from app.core.ai.signal_generator import AISignalGenerator
from app.services.telegram_bot.bot import telegram_bot_service

router = APIRouter()
signal_generator = AISignalGenerator()

@router.get("/", response_model=List[SignalResponse])
async def get_signals(
    symbol: Optional[str] = None,
    signal_type: Optional[SignalType] = None,
    status: Optional[SignalStatus] = None,
    limit: int = Query(50, ge=1, le=100),
    skip: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Get trading signals with optional filtering."""
    query = select(Signal).order_by(desc(Signal.created_at))
    
    # Apply filters if provided
    if symbol:
        query = query.filter(Signal.symbol == symbol)
    if signal_type:
        query = query.filter(Signal.signal_type == signal_type)
    if status:
        query = query.filter(Signal.status == status)
    
    # Apply pagination
    query = query.offset(skip).limit(limit)
    
    result = await db.execute(query)
    signals = result.scalars().all()
    
    return signals

@router.post("/generate", response_model=List[SignalResponse])
async def generate_signals(
    symbols: List[str],
    timeframes: List[str],
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_superuser),
):
    """Generate new signals using the AI model (admin only)."""
    signals = await signal_generator.generate_signals(symbols, timeframes)
    
    # Save signals to database
    for signal in signals:
        db.add(signal)
    
    await db.commit()
    
    # Refresh signals to get IDs and other generated fields
    for signal in signals:
        await db.refresh(signal)
    
    return signals

@router.post("/broadcast", status_code=status.HTTP_202_ACCEPTED)
async def broadcast_signal(
    signal_id: uuid.UUID,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_superuser),
):
    """Broadcast a signal to all subscribed users via Telegram."""
    # Get signal
    result = await db.execute(select(Signal).filter(Signal.id == signal_id))
    signal = result.scalar_one_or_none()
    
    if not signal:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Signal not found"
        )
    
    # Get all active users with Telegram chat ID
    query = select(User).join(Subscription).filter(
        User.telegram_chat_id.isnot(None),
        User.is_active == True,
        Subscription.status == SubscriptionStatus.ACTIVE,
        Subscription.end_date > datetime.utcnow()
    )
    
    result = await db.execute(query)
    users = result.scalars().all()
    
    if not users:
        return {"message": "No users to broadcast to"}
    
    # Add task to send signals to all users
    background_tasks.add_task(broadcast_signal_to_users, signal, users)
    
    return {"message": f"Broadcasting signal to {len(users)} users"}

@router.put("/{signal_id}", response_model=SignalResponse)
async def update_signal(
    signal_id: uuid.UUID,
    signal_update: SignalUpdate,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_superuser),
):
    """Update a signal's status and other fields."""
    result = await db.execute(select(Signal).filter(Signal.id == signal_id))
    signal = result.scalar_one_or_none()
    
    if not signal:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Signal not found"
        )
    
    # Update signal fields from the request
    status_changed = False
    previous_status = signal.status
    
    for field, value in signal_update.dict(exclude_unset=True).items():
        if field == "status" and value != previous_status:
            status_changed = True
        setattr(signal, field, value)
    
    # If status changed to TP_HIT or SL_HIT, set close time
    if status_changed and signal.status in [SignalStatus.TP_HIT, SignalStatus.SL_HIT]:
        signal.close_time = datetime.utcnow()
        
        # Calculate profit/loss
        if signal.status == SignalStatus.TP_HIT:
            if signal.signal_type == SignalType.BUY:
                signal.profit_loss = (signal.take_profit - signal.entry_price) / signal.entry_price * 100
            else:
                signal.profit_loss = (signal.entry_price - signal.take_profit) / signal.entry_price * 100
        elif signal.status == SignalStatus.SL_HIT:
            if signal.signal_type == SignalType.BUY:
                signal.profit_loss = (signal.stop_loss - signal.entry_price) / signal.entry_price * 100
            else:
                signal.profit_loss = (signal.entry_price - signal.stop_loss) / signal.entry_price * 100
    
    await db.commit()
    await db.refresh(signal)
    
    # If status changed to TP_HIT or SL_HIT, notify users
    if status_changed and signal.status in [SignalStatus.TP_HIT, SignalStatus.SL_HIT]:
        # Get subscribed users
        query = select(User).join(Subscription).filter(
            User.telegram_chat_id.isnot(None),
            User.is_active == True,
            Subscription.status == SubscriptionStatus.ACTIVE,
            Subscription.end_date > datetime.utcnow()
        )
        
        result = await db.execute(query)
        users = result.scalars().all()
        
        # Send notification to users
        background_tasks.add_task(send_signal_updates, signal, users)
    
    return signal

@router.delete("/{signal_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_signal(
    signal_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_superuser),
):
    """Delete a signal."""
    result = await db.execute(select(Signal).filter(Signal.id == signal_id))
    signal = result.scalar_one_or_none()
    
    if not signal:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Signal not found"
        )
    
    await db.delete(signal)
    await db.commit()
    
    return None

# Helper functions for background tasks
async def broadcast_signal_to_users(signal: Signal, users: List[User]):
    """Send signal to multiple users via Telegram."""
    for user in users:
        try:
            if user.telegram_chat_id:
                await telegram_bot_service.send_signal(user, signal)
        except Exception as e:
            logger.error(f"Error sending signal to user {user.id}: {str(e)}")

async def send_signal_updates(signal: Signal, users: List[User]):
    """Send signal updates (TP/SL hits) to users."""
    for user in users:
        try:
            if user.telegram_chat_id:
                await telegram_bot_service.send_signal_update(user, signal, signal.status)
        except Exception as e:
            logger.error(f"Error sending signal update to user {user.id}: {str(e)}")
