"""
Database Configuration and Session Management

Handles SQLAlchemy engine creation, session management,
and database connection health checks.
"""

import logging
from typing import Generator
from contextlib import contextmanager

from sqlalchemy import create_engine, event, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool

from app.config import settings

logger = logging.getLogger(__name__)

# Create SQLAlchemy engine with connection pooling
engine = create_engine(
    settings.DATABASE_URL,
    poolclass=QueuePool,
    pool_size=settings.DATABASE_POOL_SIZE,
    max_overflow=settings.DATABASE_MAX_OVERFLOW,
    pool_timeout=settings.DATABASE_POOL_TIMEOUT,
    pool_pre_ping=True,  # Verify connections before using
    echo=settings.DATABASE_ECHO,
)

# Session factory
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
    expire_on_commit=False,
)

# Base class for all models
Base = declarative_base()


# Event listeners for connection management
@event.listens_for(engine, "connect")
def receive_connect(dbapi_conn, connection_record):
    """Log new database connections"""
    logger.debug("New database connection established")


@event.listens_for(engine, "close")
def receive_close(dbapi_conn, connection_record):
    """Log database connection closures"""
    logger.debug("Database connection closed")


def get_db() -> Generator[Session, None, None]:
    """
    Dependency function to get database session.
    
    Yields a database session and ensures proper cleanup.
    Used as a FastAPI dependency.
    
    Yields:
        Session: SQLAlchemy database session
    """
    db = SessionLocal()
    try:
        yield db
    except Exception as e:
        logger.error(f"Database session error: {e}")
        db.rollback()
        raise
    finally:
        db.close()


@contextmanager
def get_db_context():
    """
    Context manager for database sessions.
    
    Use this when you need a database session outside of FastAPI dependency injection.
    
    Example:
        with get_db_context() as db:
            user = db.query(User).first()
    
    Yields:
        Session: SQLAlchemy database session
    """
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception as e:
        logger.error(f"Database context error: {e}")
        db.rollback()
        raise
    finally:
        db.close()


async def check_database_health() -> bool:
    """
    Check database connectivity and health.
    
    Returns:
        bool: True if database is healthy, False otherwise
    """
    try:
        with get_db_context() as db:
            # Simple query to verify connection
            db.execute(text("SELECT 1"))
        return True
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return False


def init_db():
    """
    Initialize database tables.
    
    Creates all tables defined in models.
    Should only be used for development or initial setup.
    In production, use Alembic migrations instead.
    """
    logger.info("Initializing database tables")
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables created successfully")


def drop_db():
    """
    Drop all database tables.
    
    WARNING: This is destructive and should only be used in development.
    """
    logger.warning("Dropping all database tables")
    Base.metadata.drop_all(bind=engine)
    logger.info("Database tables dropped successfully")
