# __init__.py
from .auth import router as auth_router
from .detection import router as detection_router
from .monitoring import router as monitoring_router

__all__ = [
    "auth_router",
    "detection_router",
    "monitoring_router"
]