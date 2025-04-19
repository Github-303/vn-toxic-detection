"""User model for the application."""
from datetime import datetime
from typing import Optional
from uuid import UUID

from sqlalchemy import Boolean, Column, DateTime, String, func
from sqlalchemy.dialects.postgresql import UUID as PostgresUUID
from sqlalchemy.orm import relationship

from app.models.database import Base

class User(Base):
    """User model."""
    __tablename__ = "users"
    __table_args__ = {'extend_existing': True}

    id = Column(PostgresUUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid())
    username = Column(String(50), unique=True, nullable=False)
    email = Column(String(100), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    role = Column(String(20), nullable=False, server_default="user")
    is_active = Column(Boolean, server_default="true")
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    last_login = Column(DateTime(timezone=True), nullable=True)

    # Relationships
    comments = relationship("Comment", back_populates="user", cascade="all, delete-orphan")
    logs = relationship("Log", back_populates="user", cascade="all, delete-orphan")
    social_tokens = relationship("UserSocialToken", back_populates="user", cascade="all, delete-orphan")
    api_calls = relationship("SocialMediaAPICall", back_populates="user")
    rate_limits = relationship("RateLimitTracking", back_populates="user") 

"""User schemas."""
from pydantic import BaseModel, EmailStr, constr

class UserCreate(BaseModel):
    """Schema for user creation."""
    username: constr(min_length=3, max_length=50)
    email: EmailStr
    password: constr(min_length=8)
    password_confirm: constr(min_length=8)

    class Config:
        """Pydantic config."""
        from_attributes = True