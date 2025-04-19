"""
Machine Learning service for comment analysis.
"""
from typing import Dict, Any, Literal, Optional
import logging
import torch
import os
import warnings
from pathlib import Path
from app.ml.models.transformer_models import TransformerModelManager
from app.ml.data.preprocessing import TextPreprocessor

# Tắt cảnh báo từ huggingface_hub
warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub")

logger = logging.getLogger(__name__)

ModelType = Literal["bert", "bert4news", "cnn", "grn", "ltsm", "phobert"]

class MLService:
    """Service for handling machine learning predictions."""
    
    AVAILABLE_MODELS = {
        "bert": {
            "base_model": "vinai/phobert-base",
            "config_file": "config_bert.json",
            "model_file": "model_bert.safetensors",
            "args_file": "training_args_bert.bin"
        },
        "bert4news": {
            "base_model": "NlpHUST/vibert4news-base-cased",
            "config_file": "config_bert4news.json",
            "model_file": "model_bert4news.safetensors",
            "args_file": "training_args_bert4news.bin"
        },
        "cnn": {
            "base_model": "vinai/phobert-base",
            "config_file": "config_cnn.json",
            "model_file": "model_cnn.safetensors",
            "args_file": "training_args_cnn.bin"
        },
        "grn": {
            "base_model": "vinai/phobert-base",
            "config_file": "config_grn.json",
            "model_file": "model_grn.safetensors",
            "args_file": "training_args_grn.bin"
        },
        "ltsm": {
            "base_model": "vinai/phobert-base",
            "config_file": "config_ltsm.json",
            "model_file": "model_ltsm.safetensors",
            "args_file": "training_args_ltsm.bin"
        },
        "phobert": {
            "base_model": "vinai/phobert-base",
            "config_file": "config_phobert.json",
            "model_file": "model_phobert.safetensors",
            "args_file": "training_args_phobert.bin"
        }
    }
    
    def __init__(self, model_type: ModelType = None, mock_if_missing: bool = True):
        """
        Initialize ML service.
        
        Args:
            model_type: Type of model to use (bert, bert4news, cnn, grn, ltsm, phobert)
            mock_if_missing: If True, use a mock service when model files are missing
        """
        self.is_mock = False
        
        # Kiểm tra xem có thư mục model nào tồn tại không
        if model_type is None:
            model_type = self._find_available_model()
            
        self.model_type = model_type
        
        if model_type not in self.AVAILABLE_MODELS:
            raise ValueError(f"Model type {model_type} not supported. Available models: {list(self.AVAILABLE_MODELS.keys())}")
            
        model_config = self.AVAILABLE_MODELS[model_type]
        
        # Get absolute path to model directory
        current_dir = Path(__file__).parent
        project_root = current_dir.parent.parent
        self.model_dir = project_root / "ml" / "h5_safetensors" / model_type
        
        # Create directory if it doesn't exist
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Check model files exist
        config_path = self.model_dir / model_config["config_file"]
        model_path = self.model_dir / model_config["model_file"]
        training_args_path = self.model_dir / model_config["args_file"]
        
        required_files = [config_path, model_path, training_args_path]
        missing_files = [str(p) for p in required_files if not p.exists()]
        
        if missing_files:
            message = (
                f"Missing model files for {model_type}:\n"
                f"- Missing: {', '.join(missing_files)}\n"
                f"- Directory: {self.model_dir}\n"
                f"Ensure you have the correct model files in the appropriate directory."
            )
            logger.warning(message)
            
            if mock_if_missing:
                logger.info("Using mock ML service instead")
                self.is_mock = True
                self.preprocessor = TextPreprocessor(max_length=256)
                self.model = None
                self.model_manager = None
                return
            else:
                raise RuntimeError(message)
        
        # Initialize preprocessor
        self.preprocessor = TextPreprocessor(max_length=256)
        
        try:
            # Initialize model manager with base model and load weights
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                self.model_manager = TransformerModelManager(
                    model_name=model_config["base_model"],
                    num_labels=4,
                    model_dir=str(self.model_dir)
                )
            
            logger.info(f"Loading {model_type} model from {self.model_dir}")
            self.model = self.model_manager.load_model(
                model_path=str(model_path),
                config_path=str(config_path),
                training_args_path=str(training_args_path)
            )
            self.model.eval()
        except Exception as e:
            if mock_if_missing:
                logger.error(f"Error loading model: {str(e)}")
                logger.info("Using mock ML service instead")
                self.is_mock = True
                self.model = None
                self.model_manager = None
            else:
                raise
        
    def _find_available_model(self) -> ModelType:
        """
        Find the first available model with existing files.
        
        Returns:
            The first model type that has files available
        """
        current_dir = Path(__file__).parent
        project_root = current_dir.parent.parent
        
        # Kiểm tra các thư mục model
        for model_type in self.AVAILABLE_MODELS.keys():
            model_dir = project_root / "ml" / "h5_safetensors" / model_type
            if model_dir.exists():
                config = self.AVAILABLE_MODELS[model_type]
                files_exist = all(
                    (model_dir / file).exists() 
                    for file in [config["config_file"], config["model_file"], config["args_file"]]
                )
                if files_exist:
                    logger.info(f"Found existing model: {model_type}")
                    return model_type
        
        # Nếu không tìm thấy, trả về mặc định
        return "bert"
    
    @property
    def model_version(self) -> str:
        """Get model version."""
        if self.is_mock:
            return "mock_model_v1.0"
        return f"{self.model_type}_v1.0"
        
    async def predict(self, text: str) -> Dict[str, Any]:
        """
        Predict toxicity level for a comment.
        
        Args:
            text: Comment text to analyze
            
        Returns:
            Dictionary containing prediction results
        """
        try:
            # Preprocess text and get features
            preprocessed_text, features = self.preprocessor.preprocess(
                text,
                tokenized=True,
                lowercased=True,
                remove_emoji=True,
                remove_stopwords=True,
                return_features=True
            )
            
            # If using mock service, return default values
            if self.is_mock:
                logger.info("Using mock predictions (model files missing)")
                
                # Simple heuristic for mock detection
                prediction_code = 0  # Default to safe
                if any(word in preprocessed_text.lower() for word in ["bad", "hate", "stupid", "dumb"]):
                    prediction_code = 1  # toxic
                
                # Map prediction to label
                label_map = {
                    0: "safe",
                    1: "toxic", 
                    2: "hate",
                    3: "offensive"
                }
                
                return {
                    "label": label_map[prediction_code],
                    "score": 0.9,
                    "preprocessed_text": preprocessed_text,
                    "prediction_code": prediction_code,
                    "spam_features": features,
                    "model_type": "mock_" + self.model_type,
                    "is_mock": True
                }
            
            # Get prediction from real model
            prediction_code = self.model_manager.predict_single(preprocessed_text)
            
            # Adjust prediction based on spam features
            is_potential_spam = (
                features['urls'] > 2 or
                features['emails'] > 1 or
                features['phones'] > 1 or
                features['special_chars'] > 10 or
                features['repeated_chars'] > 5
            )
            
            # Map prediction to label
            label_map = {
                0: "safe",
                1: "toxic", 
                2: "hate",
                3: "offensive"
            }
            
            # If potential spam is detected, override prediction
            if is_potential_spam and prediction_code == 0:
                prediction_code = 1  # Mark as toxic
            
            return {
                "label": label_map[prediction_code],
                "score": 1.0,  # Score will be added in future updates
                "preprocessed_text": preprocessed_text,
                "prediction_code": prediction_code,
                "spam_features": features,
                "model_type": self.model_type,
                "is_mock": False
            }
            
        except Exception as e:
            logger.error(f"Error in predict: {str(e)}")
            raise 