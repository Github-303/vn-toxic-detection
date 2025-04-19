"""Database utilities."""
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from contextlib import asynccontextmanager
from typing import AsyncGenerator
from sqlalchemy import text
import logging

from app.config.settings import settings
from app.models.database import Base, get_db

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create async engine for async operations
async_engine = create_async_engine(
    settings.DATABASE_URL,
    echo=True,
    future=True,
    pool_pre_ping=True  # Enable connection health checks
)

# Create async session factory
AsyncSessionLocal = sessionmaker(
    async_engine,
    class_=AsyncSession,
    expire_on_commit=False
)

@asynccontextmanager
async def get_async_db() -> AsyncGenerator[AsyncSession, None]:
    """Get async database session."""
    async with AsyncSessionLocal() as session:
        try:
            # Test the connection
            await session.execute(text('SELECT 1'))
            yield session
            await session.commit()
        except Exception as e:
            logger.error(f"Database error: {str(e)}")
            await session.rollback()
            raise
        finally:
            await session.close()

async def init_db():
    """Initialize database and create tables."""
    try:
        async with async_engine.begin() as conn:
            # Try to create vector extension
            try:
                await conn.execute(text('CREATE EXTENSION IF NOT EXISTS vector'))
                logger.info("Vector extension created or already exists")
            except Exception as e:
                logger.error(f"Error creating vector extension: {str(e)}")
                
            # Create tables
            await conn.run_sync(Base.metadata.create_all)
            logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Error initializing database: {str(e)}")
        raise 