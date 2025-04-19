"""User schemas."""
from datetime import datetime
from typing import Optional
from uuid import UUID
from pydantic import BaseModel, EmailStr, constr, ConfigDict

class UserBase(BaseModel):
    """Base user schema."""
    model_config = ConfigDict(from_attributes=True)
    
    username: constr(min_length=3, max_length=50)
    email: EmailStr
    is_active: bool = True
    role: str = "user"

class UserCreate(UserBase):
    """Schema for user creation."""
    model_config = ConfigDict(from_attributes=True)
    
    password: constr(min_length=8)
    password_confirm: constr(min_length=8)

class UserUpdate(BaseModel):
    """Schema for user updates."""
    model_config = ConfigDict(from_attributes=True)
    
    username: Optional[constr(min_length=3, max_length=50)] = None
    email: Optional[EmailStr] = None
    password: Optional[constr(min_length=8)] = None
    is_active: Optional[bool] = None
    role: Optional[str] = None

class UserResponse(UserBase):
    """Schema for user responses."""
    model_config = ConfigDict(from_attributes=True)
    
    id: UUID
    created_at: datetime
    last_login: Optional[datetime] = None 