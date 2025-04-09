from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, EmailStr, UUID4
from app.core.database import db

class UserBase(BaseModel):
    """Base User Model"""
    email: EmailStr
    username: str
    is_active: bool = True
    is_superuser: bool = False
    telegram_id: Optional[str] = None
    telegram_chat_id: Optional[str] = None
    discord_id: Optional[str] = None

class UserCreate(UserBase):
    """User Creation Model"""
    password: str

class UserDB(UserBase):
    """User Database Model"""
    id: UUID4
    hashed_password: str
    created_at: datetime
    updated_at: datetime

    class Config:
        orm_mode = True

class UserResponse(UserBase):
    """User Response Model"""
    id: UUID4
    created_at: datetime
    updated_at: datetime

    class Config:
        orm_mode = True

class UserService:
    """User Database Operations"""
    TABLE = "users"

    @staticmethod
    async def create(user_data: dict) -> dict:
        """Create a new user"""
        user_data['created_at'] = datetime.utcnow()
        user_data['updated_at'] = datetime.utcnow()
        return await db.execute(UserService.TABLE, 'insert', user_data)

    @staticmethod
    async def get_by_email(email: str) -> Optional[dict]:
        """Get user by email"""
        result = await db.execute(UserService.TABLE, 'select', filters={'email': email})
        return result.data[0] if result and result.data else None

    @staticmethod
    async def get_by_id(user_id: str) -> Optional[dict]:
        """Get user by ID"""
        result = await db.execute(UserService.TABLE, 'select', filters={'id': user_id})
        return result.data[0] if result and result.data else None

    @staticmethod
    async def update(user_id: str, update_data: dict) -> dict:
        """Update user data"""
        update_data['updated_at'] = datetime.utcnow()
        return await db.execute(UserService.TABLE, 'update',
                              data=update_data,
                              filters={'id': user_id})

    @staticmethod
    async def delete(user_id: str) -> dict:
        """Delete user"""
        return await db.execute(UserService.TABLE, 'delete', filters={'id': user_id})

    @staticmethod
    async def list_users(filters: dict = None) -> List[dict]:
        """List users with optional filters"""
        result = await db.execute(UserService.TABLE, 'select', filters=filters)
        return result.data if result else []
