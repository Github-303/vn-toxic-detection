"""CommentVector model for storing embeddings."""
from datetime import datetime
from uuid import UUID, uuid4
from sqlalchemy import Column, String, DateTime, ARRAY, Float, ForeignKey
from sqlalchemy.dialects.postgresql import UUID as PostgresUUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from pgvector.sqlalchemy import Vector

from app.models.database import Base

class CommentVector(Base):
    """CommentVector model."""
    __tablename__ = "comment_vectors"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid4()))
    embedding = Column(ARRAY(Float), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    model_version = Column(String(50), nullable=True, default="default")

    # Relationships
    comment = relationship("Comment", back_populates="vector", uselist=False) 