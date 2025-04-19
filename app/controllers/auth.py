from datetime import datetime, timedelta
from typing import Optional
from jose import jwt
from passlib.context import CryptContext
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.config.settings import settings
from app.models.user import User
from app.utils.database import get_async_db
from app.utils.auth import verify_password, get_password_hash

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

class AuthController:
    @staticmethod
    def verify_password(plain_password: str, hashed_password: str) -> bool:
        return pwd_context.verify(plain_password, hashed_password)

    @staticmethod
    def get_password_hash(password: str) -> str:
        return pwd_context.hash(password)

    @staticmethod
    def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
        
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(
            to_encode,
            settings.JWT_SECRET_KEY,
            algorithm=settings.JWT_ALGORITHM
        )
        return encoded_jwt

    @staticmethod
    async def register_user(
        db: AsyncSession,
        username: str,
        email: str,
        password: str
    ) -> User:
        """Register a new user."""
        # Check if email already exists
        result = await db.execute(
            select(User).filter(User.email == email)
        )
        if result.scalar_one_or_none() is not None:
            raise ValueError("Email already registered")
        
        # Check if username already exists
        result = await db.execute(
            select(User).filter(User.username == username)
        )
        if result.scalar_one_or_none() is not None:
            raise ValueError("Username already taken")
        
        # Create new user
        user = User(
            username=username,
            email=email,
            password_hash=get_password_hash(password),
            role="user",
            is_active=True
        )
        
        db.add(user)
        await db.commit()
        await db.refresh(user)
        
        return user

    @staticmethod
    async def authenticate_user(
        db: AsyncSession,
        email: str,
        password: str
    ) -> Optional[User]:
        """Authenticate a user."""
        result = await db.execute(
            select(User).filter(User.email == email)
        )
        user = result.scalar_one_or_none()
        
        if user is None:
            return None
        
        if not verify_password(password, user.password_hash):
            return None
        
        if not user.is_active:
            return None
        
        # Update last login
        user.last_login = datetime.utcnow()
        await db.commit()
        await db.refresh(user)
        
        return user 