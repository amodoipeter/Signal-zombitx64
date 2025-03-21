from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from typing import AsyncGenerator

from app.core.config import settings

# Async engine for main application usage
async_engine = create_async_engine(settings.DATABASE_URI.replace('postgresql://', 'postgresql+asyncpg://'))
AsyncSessionLocal = sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)

# Sync engine for compatibility with some libraries
engine = create_engine(settings.DATABASE_URI)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

async def init_db():
    """Initialize database, creating tables if they don't exist."""
    async with async_engine.begin() as conn:
        # await conn.run_sync(Base.metadata.drop_all)  # Uncomment for clean start
        await conn.run_sync(Base.metadata.create_all)
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Dependency for getting async DB session."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()
            await session.close()
