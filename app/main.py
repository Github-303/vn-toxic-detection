"""
Main application module
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator
import logging

from app.config.settings import settings
from app.presenters.routers import auth_router, detection_router, monitoring_router
from app.controllers.ml_controller import MLController
from app.utils.database import init_db

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title=settings.PROJECT_NAME,
    description=settings.API_DESCRIPTION,
    version=settings.API_VERSION,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    openapi_tags=[
        {
            "name": "Authentication",
            "description": "User registration and login"
        },
        {
            "name": "AI Detection",
            "description": "Comment classification endpoints"
        },
        {
            "name": "Monitoring",
            "description": "System monitoring endpoints"
        }
    ]
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize ML controller
try:
    ml_controller = MLController()
except Exception as e:
    logger.error(f"Error initializing ML controller: {str(e)}")
    ml_controller = None

# Include routers
app.include_router(auth_router, prefix=settings.API_V1_STR)
app.include_router(detection_router, prefix=settings.API_V1_STR)
app.include_router(monitoring_router, prefix=settings.API_V1_STR)

# Prometheus metrics
Instrumentator().instrument(app).expose(app)

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    try:
        # Initialize database
        await init_db()
        logger.info("Database initialized successfully")
        
        # Initialize ML controller if not already initialized
        global ml_controller
        if ml_controller is None:
            ml_controller = MLController()
            logger.info("ML controller initialized successfully")
            
    except Exception as e:
        logger.error(f"Error during startup: {str(e)}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    # Add cleanup code here
    pass

@app.get("/")
async def root():
    """Root endpoint"""
    try:
        if ml_controller is None:
            raise HTTPException(
                status_code=500,
                detail="ML controller not initialized"
            )
        
        model_info = await ml_controller.get_model_info()
        
        return {
            "project": settings.PROJECT_NAME,
            "version": settings.API_VERSION,
            "model": {
                "type": settings.MODEL_TYPE,
                "name": settings.MODEL_NAME,
                "status": model_info["status"]
            },
            "api": {
                "version": settings.API_V1_STR,
                "docs": "/docs",
                "redoc": "/redoc"
            }
        }
    except Exception as e:
        logger.error(f"Error in root endpoint: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )