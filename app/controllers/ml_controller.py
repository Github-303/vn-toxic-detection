# app/controllers/ml_controller.py
"""
Controller for machine learning endpoints
"""
import os
import logging
from typing import Dict, List, Any, Optional
from fastapi import HTTPException
import asyncio
from functools import lru_cache

from app.ml.service import ViHSDService
from app.config.settings import settings

# Initialize logger
logger = logging.getLogger(__name__)

class MLController:
    """
    Controller for machine learning operations
    """
    
    def __init__(self):
        """Initialize ML controller"""
        self._service = None
        self._lock = asyncio.Lock()
        self._initialize_service()
    
    def _initialize_service(self) -> None:
        """Initialize the ML service"""
        try:
            self._service = ViHSDService()
            logger.info("ML service initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize ML service: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"ML service initialization failed: {str(e)}"
            )

    async def analyze_text(self, text: str) -> Dict[str, Any]:
        """
        Analyze text for toxic content
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary containing analysis results
        """
        async with self._lock:
            try:
                if not self._service:
                    self._initialize_service()
                
                result = self._service.predict(text)
                
                if not result["success"]:
                    raise HTTPException(
                        status_code=500,
                        detail=f"Prediction failed: {result.get('error', 'Unknown error')}"
                    )
                
                return {
                    "text": result["text"],
                    "label": result["label"],
                    "preprocessed_text": result["preprocessed_text"],
                    "prediction_code": result["prediction"]
                }
                
            except Exception as e:
                logger.error(f"Error in analyze_text: {str(e)}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Analysis failed: {str(e)}"
                )

    async def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        return {
            "model_type": "transformer",
            "model_name": "phobert",
            "labels": ViHSDService.LABEL_NAMES,
            "status": "loaded" if self._service else "not_loaded"
        }

    def predict(self, text: str) -> dict:
        """
        Predict toxicity of a text
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary containing prediction results
        """
        if not self._service:
            raise HTTPException(
                status_code=500,
                detail="ML service not initialized"
            )
        
        try:
            return self._service.predict(text)
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Prediction failed: {str(e)}"
            )    
    def batch_predict(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Predict hate speech categories for multiple texts
        
        Args:
            texts: List of input texts
        
        Returns:
            List of prediction results
        """
        if self._service is None:
            return [{
                "success": False,
                "error": "ML service not initialized"
            }]
        
        try:
            results = self._service.batch_predict(texts)
            return results
        
        except Exception as e:
            logger.error(f"Error making batch prediction: {str(e)}")
            return [{
                "success": False,
                "error": str(e)
            }]
