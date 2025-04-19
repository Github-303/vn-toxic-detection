"""Log model for activity logging."""
from datetime import datetime
from uuid import UUID

from sqlalchemy import Column, DateTime, ForeignKey, String, func
from sqlalchemy.dialects.postgresql import JSONB, UUID as PostgresUUID
from sqlalchemy.orm import relationship

from app.models.database import Base

class Log(Base):
    """Log model."""
    __tablename__ = "logs"
    __table_args__ = {'extend_existing': True}

    id = Column(PostgresUUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid())
    user_id = Column(PostgresUUID(as_uuid=True), ForeignKey("users.id"))
    action = Column(String(50), nullable=False)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    extra_data = Column(JSONB, comment="Additional metadata for the log entry")

    # Relationships
    user = relationship("User", back_populates="logs") 