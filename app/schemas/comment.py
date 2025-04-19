"""Schema definitions for comments."""
from enum import Enum
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field, ConfigDict

class ToxicityLevel(str, Enum):
    """Toxicity level of a comment."""
    SAFE = "safe"
    TOXIC = "toxic"
    HATE = "hate"
    OFFENSIVE = "offensive"

class CommentBase(BaseModel):
    """Base schema for comments."""
    content: str = Field(..., min_length=1, max_length=1000)
    platform: str = Field(default="web")

class CommentCreate(CommentBase):
    """Schema for creating a new comment."""
    model_config = ConfigDict(from_attributes=True)

class CommentResponse(BaseModel):
    """Schema for comment responses."""
    id: str
    user_id: Optional[str] = None
    content: str
    platform: str
    toxicity_level: str
    toxicity_score: float
    preprocessed_text: Optional[str] = None
    detected_at: datetime
    created_at: datetime
    
    model_config = ConfigDict(from_attributes=True)

class CommentUpdate(BaseModel):
    """Schema for updating comment information."""
    content: Optional[str] = Field(None, min_length=1, max_length=1000)
    platform: Optional[str] = None
    toxicity_level: Optional[str] = None
    toxicity_score: Optional[float] = None
    
    model_config = ConfigDict(from_attributes=True) 