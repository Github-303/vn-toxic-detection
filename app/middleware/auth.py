from fastapi import Request, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import jwt, JWTError
from typing import Optional
from sqlalchemy import select

from app.config.settings import settings
from app.models.user import User
from app.utils.database import get_db

class AuthMiddleware(HTTPBearer):
    def __init__(self, auto_error: bool = True):
        super().__init__(auto_error=auto_error)

    async def __call__(self, request: Request) -> Optional[User]:
        credentials: HTTPAuthorizationCredentials = await super().__call__(request)
        
        if not credentials:
            if self.auto_error:
                raise HTTPException(status_code=403, detail="Missing token")
            return None
        
        try:
            token = credentials.credentials
            payload = jwt.decode(
                token,
                settings.JWT_SECRET_KEY,
                algorithms=[settings.JWT_ALGORITHM]
            )
            user_id = payload.get("sub")
            if not user_id:
                raise HTTPException(status_code=403, detail="Invalid token")
            
            async with get_db() as db:
                result = await db.execute(
                    select(User).filter(User.id == user_id)
                )
                user = result.scalar_one_or_none()
                
                if not user:
                    raise HTTPException(status_code=403, detail="User not found")
                
                if not user.is_active:
                    raise HTTPException(status_code=403, detail="User is inactive")
                
                request.state.user = user
                return user
            
        except JWTError:
            raise HTTPException(status_code=403, detail="Invalid token") 