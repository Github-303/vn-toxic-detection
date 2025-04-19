"""User model for the application."""
from datetime import datetime
from typing import Optional
from uuid import UUID
import uuid

from sqlalchemy import Boolean, Column, DateTime, String, func, ForeignKey
from sqlalchemy.dialects.postgresql import UUID as PostgresUUID
from sqlalchemy.orm import relationship

from app.models.database import Base

class User(Base):
    """User model."""
    __tablename__ = "users"
    __table_args__ = {'extend_existing': True}

    id = Column(PostgresUUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid())
    username = Column(String(50), unique=True, nullable=False)
    email = Column(String(255), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    role = Column(String(20), default="user")
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    last_login = Column(DateTime(timezone=True), nullable=True)

    # Relationships
    comments = relationship("Comment", back_populates="user")
    logs = relationship("Log", back_populates="user", cascade="all, delete-orphan")
    social_tokens = relationship("UserSocialToken", back_populates="user", cascade="all, delete-orphan")
    api_calls = relationship("SocialMediaAPICall", back_populates="user")
    rate_limits = relationship("RateLimitTracking", back_populates="user", foreign_keys="RateLimitTracking.user_id") 
