# app/ml/service.py
"""
Vietnamese Hate Speech Detection Service
"""
from typing import Dict, Any, Optional
import logging
from .models.model_manager import ModelManager
from .data.preprocessing import TextPreprocessor
import torch

logger = logging.getLogger(__name__)

class ViHSDService:
    """Service for Vietnamese Hate Speech Detection"""
    
    def __init__(self, model_name: str = "phobert"):
        """
        Initialize the service
        
        Args:
            model_name: Name of the model to use (default: phobert)
        """
        self.model_name = model_name.lower()
        self.model_manager = ModelManager()
        self.preprocessor = TextPreprocessor()
        
        # Load model
        self.model = self.model_manager.get_model(self.model_name)
        self.tokenizer = self.model_manager.get_tokenizer(self.model_name)
        
        logger.info(f"Initialized ViHSDService with model: {self.model_name}")
    
    def predict(self, text: str) -> Dict[str, Any]:
        """
        Predict toxicity of a text
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary containing prediction results
        """
        try:
            # Preprocess text
            cleaned_text = self.preprocessor.clean_text(text)
            
            # Tokenize
            inputs = self.tokenizer(
                cleaned_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256
            )
            
            # Get prediction
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.softmax(outputs.logits, dim=1)
            
            # Convert to probabilities
            probs = predictions.numpy()[0]
            
            # Get class names
            class_names = ["non-toxic", "toxic", "hate", "offensive"]
            
            # Format results
            result = {
                "text": text,
                "cleaned_text": cleaned_text,
                "predictions": {
                    class_name: float(prob)
                    for class_name, prob in zip(class_names, probs)
                },
                "model": self.model_name
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error in prediction: {str(e)}")
            raise