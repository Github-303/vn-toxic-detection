"""Monitoring routes."""
from fastapi import APIRouter
from datetime import datetime
import psutil
import time
from sqlalchemy import text

from app.utils.database import get_async_db

router = APIRouter()

@router.get("/health")
async def health_check():
    """Check system health."""
    # Check database connection
    db_status = "connected"
    try:
        async with get_async_db() as db:
            await db.execute(text("SELECT 1"))
    except Exception as e:
        print(f"Database connection error: {str(e)}")
        db_status = "disconnected"
    
    # Get system metrics
    process = psutil.Process()
    uptime = datetime.now() - datetime.fromtimestamp(process.create_time())
    
    return {
        "status": "OK",
        "timestamp": datetime.utcnow().isoformat(),
        "database": db_status,
        "uptime": str(uptime).split('.')[0],  # Remove microseconds
        "memory_usage": f"{process.memory_percent():.1f}%",
        "cpu_usage": f"{process.cpu_percent():.1f}%"
    }

@router.get("/metrics")
async def get_metrics():
    # This endpoint is automatically handled by prometheus_fastapi_instrumentator
    # Additional custom metrics can be added here
    pass 