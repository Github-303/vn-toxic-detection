from datetime import datetime
from typing import Dict, Any, Optional
from sqlalchemy import select
import logging

from app.models.comment import Comment
from app.models.user import User
from app.utils.database import get_db
from app.controllers.ml_controller import MLController

logger = logging.getLogger(__name__)

class DetectionController:
    def __init__(self):
        self.ml_controller = MLController()

    async def analyze_comment(
        self,
        text: str,
        user: User,
        platform: str = "web"
    ) -> Dict[str, Any]:
        """
        Analyze a comment and store the result
        
        Args:
            text: Comment text to analyze
            user: User who made the comment
            platform: Platform where the comment was made
            
        Returns:
            Analysis result with comment details
        """
        try:
            # Get prediction from ML model
            prediction = await self.ml_controller.analyze_text(text)
            
            # Prepare comment data
            comment_data = {
                "content": text,
                "user_id": user.id,
                "platform": platform,
                "label": prediction["label"],
                "prediction_code": prediction["prediction_code"],
                "preprocessed_text": prediction["preprocessed_text"],
                "detected_at": datetime.utcnow()
            }
            
            # Store in database
            async with get_db() as db:
                comment = Comment(**comment_data)
                db.add(comment)
                await db.commit()
                await db.refresh(comment)
            
            return {
                "comment_id": str(comment.id),
                "content": comment.content,
                "label": comment.label,
                "prediction_code": comment.prediction_code,
                "platform": comment.platform,
                "detected_at": comment.detected_at.isoformat(),
                "preprocessed_text": comment.preprocessed_text
            }
            
        except Exception as e:
            logger.error(f"Error in analyze_comment: {str(e)}")
            raise

    async def get_detection_stats(self, user: User) -> Dict[str, Any]:
        """
        Get detection statistics for a user
        
        Args:
            user: User to get statistics for
            
        Returns:
            Dictionary containing detection statistics
        """
        try:
            async with get_db() as db:
                # Get total comments
                total_query = select(Comment).filter(Comment.user_id == user.id)
                total_result = await db.execute(total_query)
                total_comments = len(total_result.scalars().all())
                
                # Get stats by label
                stats = {}
                for label in ["Clean", "Offensive", "Hate", "Spam"]:
                    label_query = select(Comment).filter(
                        Comment.user_id == user.id,
                        Comment.label == label
                    )
                    label_result = await db.execute(label_query)
                    stats[label.lower()] = len(label_result.scalars().all())
                
                return {
                    "total_comments": total_comments,
                    "stats": stats,
                    "user_id": str(user.id)
                }
                
        except Exception as e:
            logger.error(f"Error in get_detection_stats: {str(e)}")
            raise 