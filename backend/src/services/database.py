"""
Database Service for Physical AI & Humanoid Robotics Platform
"""
import os
from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import QueuePool
from contextlib import asynccontextmanager
from typing import AsyncGenerator
import logging

from src.models import Base
from src.utils.config import settings

# Set up logging
logger = logging.getLogger(__name__)

# Create async engine
engine = create_async_engine(
    settings.DATABASE_URL,
    poolclass=QueuePool,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,
    pool_recycle=300,
    echo=settings.DEBUG
)

# Create async session maker
AsyncSessionFactory = sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
    autocommit=False
)

@asynccontextmanager
async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Get database session for dependency injection
    """
    async with AsyncSessionFactory() as session:
        try:
            yield session
        except Exception as e:
            await session.rollback()
            logger.error(f"Database session error: {e}")
            raise
        finally:
            await session.close()


async def init_db() -> None:
    """
    Initialize database tables
    """
    try:
        async with engine.begin() as conn:
            # Create all tables
            await conn.run_sync(Base.metadata.create_all)

        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        raise


async def get_db_health() -> dict:
    """
    Get database health status
    """
    try:
        async with engine.connect() as conn:
            # Test connection by running a simple query
            result = await conn.execute("SELECT 1")
            row = result.fetchone()

            if row and row[0] == 1:
                return {
                    "status": "healthy",
                    "connection": "successful",
                    "timestamp": __import__('datetime').datetime.utcnow().isoformat()
                }
            else:
                return {
                    "status": "unhealthy",
                    "connection": "failed",
                    "timestamp": __import__('datetime').datetime.utcnow().isoformat()
                }

    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return {
            "status": "unhealthy",
            "connection": "error",
            "error": str(e),
            "timestamp": __import__('datetime').datetime.utcnow().isoformat()
        }


# Initialize the database when module is imported
async def startup_db():
    """
    Startup function to initialize database
    """
    await init_db()
    logger.info("Database service initialized")


# Export the engine and session factory for use in other modules
__all__ = [
    "engine",
    "AsyncSessionFactory",
    "get_db_session",
    "init_db",
    "get_db_health",
    "startup_db"
]