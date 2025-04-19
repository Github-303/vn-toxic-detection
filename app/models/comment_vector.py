"""CommentVector model for storing embeddings."""
from uuid import UUID

from sqlalchemy import Column, String, func
from sqlalchemy.dialects.postgresql import UUID as PostgresUUID
from sqlalchemy.orm import relationship
from pgvector.sqlalchemy import Vector

from app.models.database import Base

class CommentVector(Base):
    """CommentVector model."""
    __tablename__ = "comment_vectors"

    id = Column(PostgresUUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid())
    embedding = Column(Vector(768), nullable=False)
    model_version = Column(String(50), nullable=False)

    # Relationships
    comments = relationship("Comment", back_populates="vector") 