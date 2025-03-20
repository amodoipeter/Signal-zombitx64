from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Dict, Optional
from datetime import datetime, timedelta

from app.api.deps import get_current_active_user, get_current_superuser
from app.core.database import get_db
from app.models.user import User
from app.services.analytics.performance import performance_analytics

router = APIRouter()

@router.get("/performance/weekly")
async def get_weekly_performance(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Get weekly performance report."""
    try:
        report = await performance_analytics.get_weekly_report(db)
        return report
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating weekly report: {str(e)}"
        )

@router.get("/performance/monthly")
async def get_monthly_performance(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Get monthly performance report."""
    try:
        report = await performance_analytics.get_monthly_report(db)
        return report
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating monthly report: {str(e)}"
        )

@router.get("/performance/custom")
async def get_custom_performance(
    start_date: datetime,
    end_date: datetime = None,
    market: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Get custom performance report for specific date range."""
    if end_date is None:
        end_date = datetime.utcnow()
        
    try:
        report = await performance_analytics.calculate_win_rate(db, start_date, end_date, market)
        return report
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating custom report: {str(e)}"
        )
