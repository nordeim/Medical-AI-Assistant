"""
Health Check API Routes

Simple health monitoring endpoints.
"""

import logging
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.database import get_db, check_database_health
from app.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/health", tags=["Health"])


@router.get("")
async def health_check():
    """
    Basic health check endpoint.
    
    Returns:
        dict: Health status
    """
    return {
        "status": "healthy",
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION
    }


@router.get("/detailed")
async def detailed_health_check(db: Session = Depends(get_db)):
    """
    Detailed health check including database connectivity.
    
    Returns:
        dict: Detailed health status
    """
    # Check database
    db_healthy = await check_database_health()
    
    # TODO: Add model health check
    # TODO: Add vector store health check
    
    health_status = {
        "status": "healthy" if db_healthy else "unhealthy",
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "checks": {
            "database": "healthy" if db_healthy else "unhealthy",
            "model": "not_implemented",
            "vector_store": "not_implemented"
        }
    }
    
    return health_status


@router.get("/ready")
async def readiness_check(db: Session = Depends(get_db)):
    """
    Kubernetes readiness probe endpoint.
    
    Returns 200 if service is ready to accept traffic.
    """
    db_healthy = await check_database_health()
    
    if not db_healthy:
        logger.error("Readiness check failed: database unhealthy")
        return {"ready": False, "reason": "database_unhealthy"}, 503
    
    return {"ready": True}


@router.get("/live")
async def liveness_check():
    """
    Kubernetes liveness probe endpoint.
    
    Returns 200 if service is alive (even if not ready).
    """
    return {"alive": True}
