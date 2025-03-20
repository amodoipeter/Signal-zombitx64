from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks, Query
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Optional
import uuid

from app.api.deps import get_current_active_user, get_current_superuser
from app.core.database import get_db
from app.models.user import User
from app.models.redeem_code import RedeemCode, CodeStatus, CodeDuration
from app.schemas.redeem_code import (
    RedeemCodeCreate, 
    RedeemCode as RedeemCodeSchema, 
    RedeemCodeUpdate,
    RedeemRequest,
    RedeemCodePublic
)
from app.services.redeem.code_service import redeem_code_service

router = APIRouter()

@router.post("/codes", response_model=List[RedeemCodeSchema])
async def create_redeem_codes(
    code_create: RedeemCodeCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_superuser),
):
    """
    Create new redeem codes (admin only).
    
    - **subscription_tier**: Tier for the subscription
    - **duration_type**: Type of duration (SEVEN_DAYS, FIFTEEN_DAYS, THIRTY_DAYS, FREE_FOREVER, CUSTOM)
    - **duration_days**: Number of days (required only for CUSTOM duration)
    - **quantity**: How many codes to generate (default: 1)
    - **max_uses**: Maximum number of times each code can be used (null = unlimited)
    - **description**: Optional description for record keeping
    - **valid_from**: When codes become valid (default: now)
    - **valid_until**: When codes expire (default: never)
    """
    try:
        # Determine duration days based on duration type
        duration_days = None
        if code_create.duration_type == CodeDuration.CUSTOM:
            if not hasattr(code_create, 'duration_days') or not code_create.duration_days:
                raise ValueError("duration_days is required for CUSTOM duration type")
            duration_days = code_create.duration_days
        
        codes = await redeem_code_service.create_codes(
            db=db,
            tier=code_create.subscription_tier,
            duration_type=code_create.duration_type,
            duration_days=duration_days,
            quantity=code_create.quantity,
            max_uses=code_create.max_uses,
            description=code_create.description,
            valid_from=code_create.valid_from,
            valid_until=code_create.valid_until,
            created_by_id=current_user.id
        )
        return codes
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Error creating redeem codes: {str(e)}"
        )

@router.get("/codes", response_model=List[RedeemCodeSchema])
async def get_redeem_codes(
    status: Optional[CodeStatus] = None,
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=200),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_superuser),
):
    """
    Get all redeem codes with optional filtering (admin only).
    
    - **status**: Filter by code status
    - **skip**: Number of items to skip (pagination)
    - **limit**: Maximum number of items to return (pagination)
    """
    try:
        codes = await redeem_code_service.get_codes(
            db=db,
            skip=skip,
            limit=limit,
            status=status
        )
        return codes
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error fetching redeem codes: {str(e)}"
        )

@router.post("/redeem")
async def redeem_code(
    redeem_request: RedeemRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    Redeem a subscription code.
    
    - **code**: The redeem code to use
    """
    try:
        result = await redeem_code_service.redeem_code(
            db=db,
            code_str=redeem_request.code,
            user_id=current_user.id
        )
        return result
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error redeeming code: {str(e)}"
        )

@router.put("/codes/{code_id}", response_model=RedeemCodeSchema)
async def update_redeem_code(
    code_id: uuid.UUID,
    code_update: RedeemCodeUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_superuser),
):
    """
    Update a redeem code (admin only).
    
    - **status**: Update code status (e.g., disable a code)
    - **description**: Update description
    - **valid_until**: Update expiration date
    - **max_uses**: Update maximum number of uses
    """
    from sqlalchemy.future import select
    
    # Get the code first
    result = await db.execute(select(RedeemCode).filter(RedeemCode.id == code_id))
    code = result.scalar_one_or_none()
    
    if not code:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Redeem code not found"
        )
    
    # Update fields if provided
    for field, value in code_update.dict(exclude_unset=True).items():
        setattr(code, field, value)
    
    await db.commit()
    await db.refresh(code)
    
    return code

@router.delete("/codes/{code_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_redeem_code(
    code_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_superuser),
):
    """
    Delete a redeem code (admin only).
    
    This will permanently delete the code and its redemption history.
    """
    from sqlalchemy.future import select
    from app.models.redeem_code import RedeemHistory
    
    # Delete redemption history first
    await db.execute(
        """
        DELETE FROM redeem_history 
        WHERE code_id = :code_id
        """,
        {"code_id": code_id}
    )
    
    # Then delete the code
    result = await db.execute(
        """
        DELETE FROM redeem_codes 
        WHERE id = :code_id 
        RETURNING id
        """,
        {"code_id": code_id}
    )
    
    deleted_id = result.scalar_one_or_none()
    
    if not deleted_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Redeem code not found"
        )
    
    await db.commit()
    return None
