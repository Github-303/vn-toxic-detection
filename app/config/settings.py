"""
Configuration settings for ViHSD application
"""
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field
import os
from functools import lru_cache
from pydantic import ConfigDict
from sqlalchemy.orm import declarative_base

class Settings(BaseSettings):
    """Application settings"""
    
    # API settings
    PROJECT_NAME: str = Field(default="Vietnamese Hate Speech Detection API")
    API_DESCRIPTION: str = Field(default="API for detecting hate speech in Vietnamese text")
    API_VERSION: str = Field(default="1.0.0")
    API_V1_STR: str = Field(default="/api/v1")
    
    # Database
    DATABASE_URL: str = Field(
        default="postgresql+asyncpg://postgres:secret@localhost:5432/comment_db",
        description="Database connection URL"
    )
    SQLALCHEMY_DATABASE_URI: str = Field(
        default="postgresql://postgres:secret@localhost:5432/comment_db",
        description="SQLAlchemy database URI"
    )
    SQLALCHEMY_TRACK_MODIFICATIONS: bool = Field(
        default=False,
        description="SQLAlchemy track modifications flag"
    )
    SQLALCHEMY_ECHO: bool = Field(
        default=True,
        description="SQLAlchemy echo flag for debugging"
    )
    
    # Redis
    REDIS_URL: str = Field(
        default="redis://localhost:6379",
        description="Redis connection URL"
    )
    
    # JWT
    JWT_SECRET_KEY: str = "your-secret-key"
    JWT_ALGORITHM: str = Field(
        default="HS256",
        description="Algorithm for JWT encoding/decoding"
    )
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(
        default=30,
        description="JWT token expiration time in minutes"
    )
    
    # ML Model
    MODEL_NAME: str = Field(
        default="phobert",
        description="Name of the model (phobert, bert4news, bert, textcnn, gru, lstm)"
    )
    MODEL_VERSION: str = Field(
        default="1.0.0",
        description="Model version"
    )
    MODEL_TYPE: str = Field(
        default="transformer",
        description="Type of model (transformer or dnn)"
    )
    MODEL_DIR: str = Field(
        default="app/ml/h5_safetensors",
        description="Directory to save/load models"
    )
    
    # Resources
    VNCORENLP_PATH: str = Field(
        default=os.path.join(os.getcwd(), "resources/vncorenlp/VnCoreNLP-1.1.1.jar"),
        description="Path to VnCoreNLP jar file"
    )
    STOPWORDS_PATH: str = Field(
        default=os.path.join(os.getcwd(), "resources/vietnamese-stopwords.txt"),
        description="Path to stopwords file"
    )
    
    # Social Media
    FACEBOOK_APP_ID: Optional[str] = Field(
        default=None,
        description="Facebook App ID for social media integration"
    )
    FACEBOOK_APP_SECRET: Optional[str] = Field(
        default=None,
        description="Facebook App Secret for social media integration"
    )
    
    # Rate Limiting
    RATE_LIMIT_PER_MINUTE: int = Field(
        default=10,
        description="Maximum number of requests per minute per user"
    )
    MAX_BATCH_SIZE: int = Field(
        default=100,
        description="Maximum batch size for batch prediction"
    )
    
    # Environment
    ENVIRONMENT: str = Field(
        default="development",
        description="Execution environment (development, testing, production)"
    )
    
    # Application settings
    DEBUG: bool = False
    MAX_LENGTH: int = 256
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Set SQLAlchemy URI based on DATABASE_URL if not provided
        if not self.SQLALCHEMY_DATABASE_URI and self.DATABASE_URL:
            self.SQLALCHEMY_DATABASE_URI = self.DATABASE_URL.replace("+asyncpg", "")
    
    model_config = ConfigDict(from_attributes=True)

@lru_cache()
def get_settings() -> Settings:
    """Get cached application settings"""
    return Settings()

# Create settings instance
settings = Settings()