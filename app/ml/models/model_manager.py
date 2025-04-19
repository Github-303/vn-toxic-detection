"""
Model Manager for loading and managing different types of models
"""
from typing import Dict, Optional, Any
import os
import logging
from pathlib import Path
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from safetensors.torch import load_file
import tensorflow as tf
from tensorflow.keras.models import load_model

logger = logging.getLogger(__name__)

class ModelManager:
    """Manager for loading and handling different types of models"""
    
    MODELS_ROOT = Path("app/ml/h5_safetensors")
    
    # Model type mappings
    MODEL_TYPES = {
        "transformer": ["phobert", "bert", "bert4news"],
        "dnn": ["cnn", "lstm", "gru"]
    }
    
    def __init__(self):
        self.loaded_models: Dict[str, Any] = {}
        self.tokenizers: Dict[str, Any] = {}
    
    def get_model_path(self, model_name: str) -> Path:
        """Get the path for a specific model"""
        model_dir = self.MODELS_ROOT / model_name.lower()
        if not model_dir.exists():
            raise ValueError(f"Model directory not found: {model_dir}")
        return model_dir
    
    def is_transformer_model(self, model_name: str) -> bool:
        """Check if the model is a transformer-based model"""
        return model_name.lower() in self.MODEL_TYPES["transformer"]
    
    def load_transformer_model(self, model_name: str) -> None:
        """Load a transformer-based model from safetensor file"""
        try:
            model_name = model_name.lower()
            model_dir = self.get_model_path(model_name)
            
            # Load model from safetensor file
            model_file = model_dir / f"model_{model_name}.safetensors"
            if not model_file.exists():
                raise FileNotFoundError(f"Model file not found: {model_file}")
            
            # Get base model name for tokenizer
            if model_name == "phobert":
                base_model = "vinai/phobert-base"
            elif model_name == "bert4news":
                base_model = "vinai/bert4news-base"
            else:
                base_model = "bert-base-uncased"
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(base_model)
            self.tokenizers[model_name] = tokenizer
            
            # Create model with correct architecture
            model = AutoModelForSequenceClassification.from_pretrained(
                base_model,
                num_labels=4,  # Number of classes
                ignore_mismatched_sizes=True
            )
            
            # Load weights from safetensor file
            state_dict = load_file(str(model_file))
            model.load_state_dict(state_dict)
            
            # Set model to evaluation mode
            model.eval()
            
            self.loaded_models[model_name] = model
            logger.info(f"Successfully loaded transformer model: {model_name}")
            
        except Exception as e:
            logger.error(f"Error loading transformer model {model_name}: {str(e)}")
            raise
    
    def load_dnn_model(self, model_name: str) -> None:
        """Load a DNN-based model from h5 file"""
        try:
            model_dir = self.get_model_path(model_name)
            
            # Get model file
            if model_name.lower() == "lstm":
                model_file = "best_model_LSTM.h5"
            elif model_name.lower() == "cnn":
                model_file = "text_cnn_model.h5"
            elif model_name.lower() == "gru":
                model_file = "gru_model.h5"
            else:
                raise ValueError(f"Unknown DNN model type: {model_name}")
            
            model_path = model_dir / model_file
            if not model_path.exists():
                raise FileNotFoundError(f"Model not found: {model_path}")
            
            # Load the model
            model = load_model(str(model_path))
            self.loaded_models[model_name] = model
            
            logger.info(f"Successfully loaded DNN model: {model_name}")
            
        except Exception as e:
            logger.error(f"Error loading DNN model {model_name}: {str(e)}")
            raise
    
    def load_model(self, model_name: str) -> None:
        """Load a model by name"""
        model_name = model_name.lower()
        
        if model_name in self.loaded_models:
            logger.info(f"Model {model_name} already loaded")
            return
        
        if self.is_transformer_model(model_name):
            self.load_transformer_model(model_name)
        else:
            self.load_dnn_model(model_name)
    
    def get_model(self, model_name: str) -> Any:
        """Get a loaded model"""
        model_name = model_name.lower()
        if model_name not in self.loaded_models:
            self.load_model(model_name)
        return self.loaded_models[model_name]
    
    def get_tokenizer(self, model_name: str) -> Any:
        """Get tokenizer for a transformer model"""
        model_name = model_name.lower()
        if not self.is_transformer_model(model_name):
            raise ValueError(f"Tokenizer not available for non-transformer model: {model_name}")
        
        if model_name not in self.tokenizers:
            self.load_model(model_name)
        return self.tokenizers[model_name] 