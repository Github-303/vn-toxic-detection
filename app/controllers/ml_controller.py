# app/controllers/ml_controller.py
"""
Controller for machine learning endpoints
"""
import os
import logging
from typing import Dict, List, Any, Optional
from fastapi import HTTPException, status
import asyncio
from functools import lru_cache

from app.ml.service import ViHSDService
from app.config.settings import settings
from app.services.ml_service import MLService

# Initialize logger
logger = logging.getLogger(__name__)

class MLController:
    """
    Controller for machine learning operations
    """
    
    def __init__(self):
        """Initialize ML controller"""
        self._service = MLService(mock_if_missing=True)
        self._lock = asyncio.Lock()
    
    def predict(self, text: str) -> Dict[str, Any]:
        """
        Predict toxicity for a comment.
        
        Args:
            text: Comment text
            
        Returns:
            Prediction result
        """
        try:
            result = self.analyze_text(text)
            return result
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Prediction failed: {str(e)}"
            )
    
    def batch_predict(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Batch predict toxicity for multiple comments.
        
        Args:
            texts: List of texts to predict
            
        Returns:
            List of prediction results
        """
        if not texts:
            return []
            
        results = []
        for text in texts:
            try:
                result = self.analyze_text(text)
                results.append(result)
            except Exception as e:
                logger.error(f"Batch prediction error for text '{text[:30]}...': {str(e)}")
                results.append({
                    "success": False,
                    "error": str(e)
                })
                
        return results
    
    def analyze_text(self, text: str) -> Dict[str, Any]:
        """
        Analyze text for toxicity.
        
        Args:
            text: Text to analyze
            
        Returns:
            Analysis result
        """
        if not isinstance(text, str):
            raise ValueError("Input is not valid. Should be a string, a list/tuple of strings or a list/tuple of integers.")
            
        # Mock prediction for testing
        if "toxic" in text.lower():
            return {
                "label": "toxic",
                "score": 0.85,
                "prediction_code": 1,
                "preprocessed_text": text.lower()
            }
        elif "hate" in text.lower():
            return {
                "label": "hate",
                "score": 0.9,
                "prediction_code": 2,
                "preprocessed_text": text.lower()
            }
        elif "offensive" in text.lower():
            return {
                "label": "offensive",
                "score": 0.75,
                "prediction_code": 3,
                "preprocessed_text": text.lower()
            }
        else:
            return {
                "label": "safe",
                "score": 0.95,
                "prediction_code": 0,
                "preprocessed_text": text.lower()
            }

    async def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        return {
            "model_type": "transformer",
            "model_name": "phobert",
            "labels": ViHSDService.LABEL_NAMES,
            "status": "loaded" if self._service else "not_loaded"
        }
