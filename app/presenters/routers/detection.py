from fastapi import APIRouter, Depends, HTTPException, status
from typing import List, Any
from pydantic import BaseModel
from sqlalchemy import select

from app.controllers.detection import DetectionController
from app.middleware.auth import AuthMiddleware
from app.models.user import User
from app.models.comment import Comment
from app.schemas.comment import CommentCreate, CommentResponse
from app.db.session import get_db
from app.services.ml_service import MLService

router = APIRouter()
auth_middleware = AuthMiddleware()

# Khởi tạo service với tự động tìm mô hình phù hợp nhất
ml_service = MLService(model_type=None, mock_if_missing=True)
detection_controller = DetectionController(ml_service=ml_service)

@router.post("/comment", response_model=CommentResponse)
async def analyze_comment(
    comment_data: CommentCreate,
    current_user: User = Depends(auth_middleware)
) -> Any:
    """
    Analyze a comment for toxicity
    
    Args:
        comment_data: Comment data to analyze
        current_user: Current authenticated user
        
    Returns:
        Analyzed comment response
    """
    async with get_db() as db:
        result = await detection_controller.analyze_comment(
            db=db,
            comment=comment_data
        )
        return result

@router.get("/model-info")
async def get_model_info() -> Any:
    """
    Get information about the current ML model
    
    Returns:
        Dictionary containing model information
    """
    return {
        "model_type": ml_service.model_type,
        "is_mock": ml_service.is_mock,
        "model_version": ml_service.model_version
    }

@router.get("/history")
async def get_comment_history(
    current_user: User = Depends(auth_middleware),
    limit: int = 10,
    offset: int = 0
) -> Any:
    """
    Get comment history for the current user
    
    Args:
        current_user: Current authenticated user
        limit: Maximum number of results to return
        offset: Number of results to skip
        
    Returns:
        List of comment history records
    """
    async with get_db() as db:
        comments = await detection_controller.get_comment_history(
            db=db,
            user_id=str(current_user.id)
        )
        
        # Apply limit and offset
        paginated_comments = comments[offset:offset + limit]
        
        return [
            {
                "id": str(comment.id),
                "content": comment.content,
                "toxicity_level": comment.toxicity_level,
                "platform": comment.platform,
                "toxicity_score": comment.toxicity_score,
                "detected_at": comment.detected_at.isoformat(),
                "created_at": comment.created_at.isoformat()
            }
            for comment in paginated_comments
        ]

@router.get("/similar/{comment_id}")
async def get_similar_comments(
    comment_id: str,
    threshold: float = 0.8,
    limit: int = 10,
    current_user: User = Depends(auth_middleware)
) -> Any:
    """
    Get similar comments based on embedding similarity
    
    Args:
        comment_id: ID of the target comment
        threshold: Minimum similarity threshold
        limit: Maximum number of results to return
        current_user: Current authenticated user
        
    Returns:
        List of similar comments with similarity scores
    """
    async with get_db() as db:
        # Get the target comment
        comment_query = select(Comment).filter(Comment.id == comment_id)
        comment_result = await db.execute(comment_query)
        comment = comment_result.scalar_one_or_none()
        
        if not comment:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Comment not found"
            )
        
        # Get similar comments
        similar_comments = await detection_controller.get_similar_comments(
            db=db,
            text=comment.content,
            limit=limit,
            min_similarity=threshold
        )
        
        return [
            {
                "id": str(item["comment"].id),
                "content": item["comment"].content,
                "toxicity_level": item["comment"].toxicity_level,
                "similarity": item["similarity"],
                "detected_at": item["comment"].detected_at.isoformat()
            }
            for item in similar_comments
        ]

@router.get("/statistics")
async def get_comment_statistics(
    current_user: User = Depends(auth_middleware)
) -> Any:
    """
    Get comment statistics
    
    Args:
        current_user: Current authenticated user
        
    Returns:
        Comment statistics
    """
    async with get_db() as db:
        return await detection_controller.get_comment_statistics(db=db) 