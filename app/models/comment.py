"""Comment model for the application."""
from datetime import datetime
from uuid import UUID

from sqlalchemy import Column, DateTime, ForeignKey, SmallInteger, String, Text, func
from sqlalchemy.dialects.postgresql import UUID as PostgresUUID
from sqlalchemy.orm import relationship

from app.models.database import Base

class Comment(Base):
    """Comment model."""
    __tablename__ = "comments"

    id = Column(PostgresUUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid())
    user_id = Column(PostgresUUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    content = Column(Text, nullable=False)
    label = Column(SmallInteger, nullable=False)
    platform = Column(String(20), nullable=False)
    detected_at = Column(DateTime(timezone=True), server_default=func.now())
    vector_id = Column(PostgresUUID(as_uuid=True), ForeignKey("comment_vectors.id"))

    # Relationships
    user = relationship("User", back_populates="comments")
    vector = relationship("CommentVector", back_populates="comments") 
"""Comment schemas."""
from enum import Enum
from pydantic import BaseModel, constr

class ToxicityLevel(str, Enum):
    """Toxicity levels."""
    SAFE = "safe"
    TOXIC = "toxic"
    HATE = "hate"
    OFFENSIVE = "offensive"

class CommentCreate(BaseModel):
    """Schema for comment creation."""
    content: constr(min_length=1, max_length=1000)
    platform: str = "web"

    class Config:
        """Pydantic config."""
        from_attributes = True