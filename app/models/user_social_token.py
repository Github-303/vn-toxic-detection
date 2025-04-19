"""User social token model."""
from datetime import datetime

from sqlalchemy import Column, DateTime, ForeignKey, String, func
from sqlalchemy.dialects.postgresql import ARRAY, UUID as PostgresUUID
from sqlalchemy.orm import relationship

from app.models.database import Base

class UserSocialToken(Base):
    """User social token model for storing platform access tokens."""
    __tablename__ = "user_social_tokens"

    user_id = Column(PostgresUUID(as_uuid=True), ForeignKey("users.id"), primary_key=True)
    platform = Column(String(20), ForeignKey("platform_configs.platform"), primary_key=True)
    access_token = Column(String, nullable=False)
    refresh_token = Column(String)
    expires_at = Column(DateTime(timezone=True))
    scopes = Column(ARRAY(String), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    user = relationship("User", back_populates="social_tokens")
    platform_config = relationship("PlatformConfig", back_populates="tokens") 