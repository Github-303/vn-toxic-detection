# app/ml/models/transformer_models.py
"""
Transformer models for Vietnamese Hate Speech Detection
"""
import os
import time
import logging
from typing import Dict, Any, Tuple, Optional, List

import torch
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)

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
        self.model_dir = os.path.join(model_dir, model_name.lower())
        
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
            self.tokenizer = config['tokenizer_class'].from_pretrained(
                config['base_model'],
                **config['tokenizer_kwargs']
            )
            
            # Initialize model
            self.logger.info(f"Loading model for {self.model_name}...")
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
        
        training_time = time.time() - start_time
        self.logger.info(f"{self.model_name} training completed in {training_time:.2f} seconds")
        
        # Save model
        output_dir = trainer.args.output_dir
        trainer.save_model(output_dir)
        self.logger.info(f"Model saved to {output_dir}")
        
        return trainer
    
    def predict(
        self, 
        test_dataset, 
        trainer: Optional[Trainer] = None
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Make predictions on test dataset and calculate metrics
        """
        if trainer is None:
            if self.trainer is None:
                raise ValueError("Trainer not initialized. Call create_trainer() first")
            trainer = self.trainer
        
        self.logger.info(f"Starting {self.model_name} evaluation...")
        start_time = time.time()
        
        # Predict
        prediction_output = trainer.predict(test_dataset)
        y_pred = np.argmax(prediction_output.predictions, axis=-1)
        
        inference_time = time.time() - start_time
        avg_inference_time = inference_time / len(test_dataset)
        self.logger.info(f"{self.model_name} inference completed in {inference_time:.2f} seconds")
        self.logger.info(f"Average inference time per sample: {avg_inference_time:.4f} seconds")
        
        # Return predictions and timing information
        metrics = {
            'inference_time': inference_time,
            'avg_inference_time': avg_inference_time
        }
        
        return y_pred, metrics
    
    def predict_single(self, text: str) -> int:
        """
        Make a prediction on a single text input
        """
        if self.model is None or self.tokenizer is None:
            self.logger.error("Model or tokenizer not initialized")
            raise ValueError("Model or tokenizer not initialized")
        
        # Preprocess and tokenize text
        encoded_input = self.tokenizer(
            text,
            truncation=True,
            padding=True,
            return_tensors='pt'
        ).to(self.device)
        
        # Make prediction
        with torch.no_grad():
            output = self.model(**encoded_input)
        
        # Get predicted class
        predicted_class = torch.argmax(output.logits, dim=1).item()
        
        return predicted_class
    
    def load_model(self, model_path: Optional[str] = None):
        """
        Load a saved model
        """
        if model_path is None:
            model_path = self.model_dir
        
        try:
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
            self.model.to(self.device)
            self.logger.info(f"Model loaded from {model_path}")
        except Exception as e:
            self.logger.error(f"Error loading model from {model_path}: {str(e)}")
            raise e