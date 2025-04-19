"""Platform configuration model."""
from sqlalchemy import Column, Integer, String, ForeignKey
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.orm import relationship

from app.models.database import Base

class PlatformConfig(Base):
    """Platform configuration model for social media integrations."""
    __tablename__ = "platform_configs"

    platform = Column(String(20), primary_key=True)
    api_base_url = Column(String(255), nullable=False)
    auth_type = Column(String(20), nullable=False)
    max_comments_per_day = Column(Integer, default=100)
    max_comments_per_hour = Column(Integer, default=10)
    max_comments_per_minute = Column(Integer, default=1)
    default_scopes = Column(ARRAY(String))

    # Relationships
    tokens = relationship("UserSocialToken", back_populates="platform_config")
    api_calls = relationship("SocialMediaAPICall", back_populates="platform_config")
    rate_limits = relationship("RateLimitTracking", back_populates="platform_config") 