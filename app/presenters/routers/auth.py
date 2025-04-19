from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from datetime import timedelta
from typing import Any

from app.controllers.auth import AuthController
from app.middleware.auth import AuthMiddleware
from app.schemas.auth import Token, UserCreate, UserResponse
from app.models.user import User

router = APIRouter()
auth_middleware = AuthMiddleware()

@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register(user_data: UserCreate) -> Any:
    try:
        user = AuthController.register_user(
            username=user_data.username,
            email=user_data.email,
            password=user_data.password
        )
        return user
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

@router.post("/login", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends()) -> Any:
    user = AuthController.authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token = AuthController.create_access_token(
        data={"sub": str(user.id)}
    )
    return {"access_token": access_token, "token_type": "bearer"}

@router.get("/users/me", response_model=UserResponse)
async def get_current_user(current_user: User = Depends(auth_middleware)) -> Any:
    """Get the current authenticated user."""
    return current_user

@router.post("/refresh", response_model=Token)
async def refresh_token(current_user: UserResponse = Depends(auth_middleware)) -> Any:
    access_token = AuthController.create_access_token(
        data={"sub": str(current_user.id)}
    )
    return {"access_token": access_token, "token_type": "bearer"} 