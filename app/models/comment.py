"""Comment model for the application."""
from datetime import datetime
from typing import Optional
from uuid import UUID, uuid4
from sqlalchemy import Column, DateTime, ForeignKey, SmallInteger, String, Text, Integer, func, Float
from sqlalchemy.dialects.postgresql import UUID as PostgresUUID
from sqlalchemy.orm import relationship

from app.models.database import Base

class Comment(Base):
    """Comment model."""
    __tablename__ = "comments"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid4()))
    user_id = Column(String(36), ForeignKey("users.id"), nullable=True)
    content = Column(String(1000), nullable=False)
    toxicity_level = Column(String(20), nullable=False)
    toxicity_score = Column(Float, nullable=False, default=0.0)
    prediction_code = Column(Integer, nullable=False, default=0)
    preprocessed_text = Column(Text, nullable=True)
    platform = Column(String(50), nullable=False, default="web")
    detected_at = Column(DateTime, default=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow)
    vector_id = Column(String(36), ForeignKey("comment_vectors.id"), nullable=True)

    # Relationships
    user = relationship("User", back_populates="comments")
    vector = relationship("CommentVector", back_populates="comment") 