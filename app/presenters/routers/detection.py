from fastapi import APIRouter, Depends, HTTPException, status
from typing import List, Any
from pydantic import BaseModel
from sqlalchemy import select

from app.controllers.detection import DetectionController
from app.middleware.auth import AuthMiddleware
from app.models.user import User
from app.models.comment import Comment
from app.utils.database import get_db

router = APIRouter()
auth_middleware = AuthMiddleware()
detection_controller = DetectionController()

class CommentRequest(BaseModel):
    text: str
    platform: str = "web"

class CommentResponse(BaseModel):
    comment_id: str
    label: int
    scores: List[dict]
    timestamp: str

@router.post("/comment")
async def analyze_comment(
    request: CommentRequest,
    current_user: User = Depends(auth_middleware)
) -> Any:
    return await detection_controller.analyze_comment(
        text=request.text,
        user=current_user,
        platform=request.platform
    )

@router.get("/history")
async def get_comment_history(
    current_user: User = Depends(auth_middleware),
    limit: int = 10,
    offset: int = 0
) -> Any:
    async with get_db() as db:
        result = await db.execute(
            select(Comment)
            .filter(Comment.user_id == current_user.id)
            .order_by(Comment.detected_at.desc())
            .offset(offset)
            .limit(limit)
        )
        comments = result.scalars().all()
        
        return [
            {
                "id": str(comment.id),
                "content": comment.content,
                "label": comment.label,
                "platform": comment.platform,
                "detected_at": comment.detected_at.isoformat()
            }
            for comment in comments
        ] 