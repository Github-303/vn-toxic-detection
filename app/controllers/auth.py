from datetime import datetime, timedelta
from typing import Optional
from jose import jwt
from passlib.context import CryptContext

from app.config.settings import settings
from app.models.user import User
from app.utils.database import get_async_db

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
    async def register_user(username: str, email: str, password: str) -> User:
        async with get_async_db() as db:
            # Check if user exists
            result = await db.execute(db.query(User).filter(User.email == email))
            if result.scalar_one_or_none():
                raise ValueError("Email already registered")
            
            result = await db.execute(db.query(User).filter(User.username == username))
            if result.scalar_one_or_none():
                raise ValueError("Username already taken")
            
            # Create new user
            hashed_password = AuthController.get_password_hash(password)
            user = User(
                username=username,
                email=email,
                password_hash=hashed_password
            )
            
            db.add(user)
            await db.commit()
            await db.refresh(user)
            return user

    @staticmethod
    async def authenticate_user(email: str, password: str) -> Optional[User]:
        async with get_async_db() as db:
            result = await db.execute(db.query(User).filter(User.email == email))
            user = result.scalar_one_or_none()
            
            if not user:
                return None
            
            if not AuthController.verify_password(password, user.password_hash):
                return None
            
            # Update last login
            user.last_login = datetime.utcnow()
            await db.commit()
            
            return user 