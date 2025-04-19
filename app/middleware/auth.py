from fastapi import Request, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import jwt, JWTError
from typing import Optional
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config.settings import settings
from app.models.user import User
from app.utils.database import get_db

class AuthMiddleware(HTTPBearer):
    def __init__(self, auto_error: bool = True):
        super().__init__(auto_error=auto_error)

    async def __call__(self, request: Request) -> Optional[User]:
        auth_header = request.headers.get("Authorization")
        if not auth_header:
            raise HTTPException(
                status_code=401,
                detail="Not authenticated"
            )
        
        credentials: HTTPAuthorizationCredentials = await super().__call__(request)
        
        if not credentials:
            if self.auto_error:
                raise HTTPException(status_code=401, detail="Not authenticated")
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
                raise HTTPException(status_code=401, detail="Invalid authentication credentials")
            
            # Get database session
            session_generator = get_db()
            db = None
            try:
                db = await session_generator.__anext__()
                
                result = await db.execute(
                    select(User).filter(User.id == user_id)
                )
                user = result.scalar_one_or_none()
                
                if not user:
                    raise HTTPException(status_code=401, detail="User not found")
                
                if not user.is_active:
                    raise HTTPException(status_code=401, detail="Inactive user")
                
                request.state.user = user
                return user
            except Exception as e:
                if db is not None:
                    try:
                        await db.rollback()
                    except:
                        pass  # Ignore errors during rollback
                raise e
            finally:
                if db is not None:
                    try:
                        await db.close()
                    except:
                        pass  # Ignore errors during close
            
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=401,
                detail="Token has expired"
            )
        except JWTError:
            raise HTTPException(
                status_code=401,
                detail="Invalid authentication credentials"
            ) 