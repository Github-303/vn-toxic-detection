"""Social media API call model."""
from datetime import datetime

from sqlalchemy import Column, DateTime, ForeignKey, Integer, SmallInteger, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB, UUID as PostgresUUID
from sqlalchemy.orm import relationship

from app.models.database import Base

class SocialMediaAPICall(Base):
    """Social media API call model for tracking platform interactions."""
    __tablename__ = "social_media_api_calls"

    id = Column(PostgresUUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid())
    platform = Column(String(20), ForeignKey("platform_configs.platform"), nullable=False)
    endpoint = Column(String(255), nullable=False)
    user_id = Column(PostgresUUID(as_uuid=True), ForeignKey("users.id"))
    parameters = Column(JSONB)
    response_status = Column(SmallInteger, nullable=False)
    response_data = Column(JSONB)
    called_at = Column(DateTime(timezone=True), server_default=func.now())
    retry_count = Column(SmallInteger, default=0)
    error_message = Column(Text)
    processed = Column(SmallInteger, default=0)  # 0: pending, 1: processed, 2: failed
    processing_time = Column(Integer)  # in milliseconds

    # Relationships
    user = relationship("User", back_populates="api_calls")
    platform_config = relationship("PlatformConfig", back_populates="api_calls") 