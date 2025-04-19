# app/ml/models/transformer_models.py
"""
Transformer models for Vietnamese Hate Speech Detection
"""
import os
import time
import logging
import json
import warnings
from typing import Dict, Any, Tuple, Optional, List
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    PretrainedConfig
)
from safetensors.torch import load_file

logger = logging.getLogger(__name__)

class TransformerModelManager:
    """
    Manages transformer-based models for Vietnamese Hate Speech Detection
    """
    
    # Available model configurations
    MODEL_CONFIGS = {
        'PhoBERT': {
            'base_model': "vinai/phobert-base",
            'tokenizer_class': AutoTokenizer,
            'model_class': AutoModelForSequenceClassification,
            'tokenizer_kwargs': {'use_fast': False},
        },
        'BERT4News': {
            'base_model': "NlpHUST/vibert4news-base-cased",
            'tokenizer_class': AutoTokenizer,
            'model_class': AutoModelForSequenceClassification,
            'tokenizer_kwargs': {'use_fast': False},
        },
        'BERT': {
            'base_model': "bert-base-uncased",
            'tokenizer_class': AutoTokenizer,
            'model_class': AutoModelForSequenceClassification,
            'tokenizer_kwargs': {'use_fast': True},
        }
    }
    
    def __init__(
        self, 
        model_name: str,
        num_labels: int = 4,
        model_dir: str = 'models',
        device: Optional[str] = None
    ):
        """
        Initialize transformer model manager
        
        Args:
            model_name: Name of the model to use (PhoBERT, BERT4News, BERT)
            num_labels: Number of classification labels
            model_dir: Directory to save models
            device: Device to use for training (cpu or cuda)
        """
        self.model_name = model_name
        self.num_labels = num_labels
        self.model_dir = model_dir
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        
        # Initialize model and tokenizer
        self.tokenizer = None
        self.model = None
        self.trainer = None
        self._initialize_model_and_tokenizer()
    
    def _initialize_model_and_tokenizer(self):
        """Initialize model and tokenizer based on model name"""
        if self.model_name not in self.MODEL_CONFIGS:
            self.logger.error(f"Model {self.model_name} not found in available configurations")
            raise ValueError(f"Model {self.model_name} not found in available configurations")
        
        config = self.MODEL_CONFIGS[self.model_name]
        
        try:
            # Initialize tokenizer
            self.logger.info(f"Loading tokenizer for {self.model_name}...")
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                self.tokenizer = config['tokenizer_class'].from_pretrained(
                    config['base_model'],
                    **config['tokenizer_kwargs']
                )
            
            # Initialize model
            self.logger.info(f"Loading model for {self.model_name}...")
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                self.model = config['model_class'].from_pretrained(
                    config['base_model'],
                    num_labels=self.num_labels,
                    ignore_mismatched_sizes=True
                )
            
            # Move model to device
            self.model.to(self.device)
            
        except Exception as e:
            self.logger.error(f"Error initializing {self.model_name}: {str(e)}")
            raise e
    
    def load_model(
        self,
        model_path: str,
        config_path: Optional[str] = None,
        training_args_path: Optional[str] = None
    ) -> AutoModelForSequenceClassification:
        """
        Load model from files
        
        Args:
            model_path: Path to model weights file (.safetensors)
            config_path: Path to model config file (.json)
            training_args_path: Path to training args file (.bin)
            
        Returns:
            Loaded model
        """
        try:
            # Load config if provided
            if config_path and os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config_dict = json.load(f)
                config = PretrainedConfig(**config_dict)
                self.model = AutoModelForSequenceClassification.from_config(config)
            
            # Load model weights
            if os.path.exists(model_path):
                state_dict = load_file(model_path)
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore")
                    self.model.load_state_dict(state_dict, strict=False)
                
            # Load training args if provided
            if training_args_path and os.path.exists(training_args_path):
                training_args = torch.load(training_args_path)
                self.model.config.update(training_args)
            
            # Move model to device
            self.model.to(self.device)
            return self.model
            
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise e
    
    def predict_single(self, text: str) -> int:
        """
        Predict class for a single text
        
        Args:
            text: Input text
            
        Returns:
            Predicted class index
        """
        # Tokenize input
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get prediction
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)
            predicted = torch.argmax(probabilities, dim=1)
            
        return predicted.item()
    
    def setup_training_args(
        self,
        output_dir: Optional[str] = None,
        num_epochs: int = 2,
        batch_size: int = 16,
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01,
        warmup_steps: int = 500,
        gradient_accumulation_steps: int = 2,
        evaluation_strategy: str = "steps",
        eval_steps: int = 100,
        save_steps: int = 100,
        metric_for_best_model: str = "eval_loss",
        fp16: bool = True
    ) -> TrainingArguments:
        """
        Setup training arguments for the trainer
        """
        if output_dir is None:
            output_dir = self.model_dir
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Create training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=warmup_steps,
            weight_decay=weight_decay,
            logging_dir=f'{output_dir}/logs',
            logging_steps=100,
            eval_steps=eval_steps,
            save_steps=save_steps,
            evaluation_strategy=evaluation_strategy,
            load_best_model_at_end=True,
            metric_for_best_model=metric_for_best_model,
            report_to="tensorboard",
            learning_rate=learning_rate,
            gradient_accumulation_steps=gradient_accumulation_steps,
            fp16=fp16
        )
        
        return training_args
    
    def create_trainer(
        self,
        train_dataset,
        eval_dataset,
        training_args: Optional[TrainingArguments] = None,
        callbacks: Optional[List] = None
    ):
        """
        Create trainer for the model
        """
        if training_args is None:
            training_args = self.setup_training_args()
        
        if callbacks is None:
            callbacks = [EarlyStoppingCallback(early_stopping_patience=3)]
        
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            callbacks=callbacks
        )
        
        return self.trainer
    
    def train(self, trainer: Optional[Trainer] = None):
        """
        Train the model
        """
        if trainer is None:
            if self.trainer is None:
                raise ValueError("Trainer not initialized. Call create_trainer() first")
            trainer = self.trainer
        
        self.logger.info(f"Starting {self.model_name} training...")
        start_time = time.time()
        
        # Train model
        trainer.train()
        
        # Log training time
        end_time = time.time()
        training_time = end_time - start_time
        self.logger.info(f"Training completed in {training_time:.2f} seconds")