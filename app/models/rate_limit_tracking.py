"""Rate limit tracking model."""
from datetime import datetime

from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, func
from sqlalchemy.dialects.postgresql import UUID as PostgresUUID
from sqlalchemy.orm import relationship

from app.models.database import Base

class RateLimitTracking(Base):
    """Rate limit tracking model for API call monitoring."""
    __tablename__ = "rate_limit_tracking"

    user_id = Column(PostgresUUID(as_uuid=True), ForeignKey("users.id"), primary_key=True)
    platform = Column(String(20), ForeignKey("platform_configs.platform"), primary_key=True)
    window_start = Column(DateTime(timezone=True), primary_key=True, server_default=func.now())
    request_count = Column(Integer, default=0)
    last_request_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # Relationships
    user = relationship("User", back_populates="rate_limits")
    platform_config = relationship("PlatformConfig", back_populates="rate_limits") 