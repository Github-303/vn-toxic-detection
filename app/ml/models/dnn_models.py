# app/ml/models/dnn_models.py
"""
Deep Neural Network models for Vietnamese Hate Speech Detection
"""
import os
import time
import logging
import numpy as np
from typing import Dict, Any, Tuple, Optional, List, Callable

# TensorFlow and Keras imports
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Dense, Embedding, Conv2D, MaxPool2D,
    Reshape, Flatten, Dropout, Concatenate,
    Bidirectional, GRU, LSTM, GlobalMaxPooling1D,
    GlobalAveragePooling1D, SpatialDropout1D,
    concatenate
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import (
    ModelCheckpoint, 
    CSVLogger, 
    EarlyStopping,
    TensorBoard
)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


class DNNModelManager:
    """
    Manages DNN-based models for Vietnamese Hate Speech Detection
    """
    
    MODEL_BUILDERS = {
        'TextCNN': 'create_text_cnn_model',
        'GRU': 'create_gru_model',
        'LSTM': 'create_lstm_model'
    }
    
    def __init__(
        self,
        model_name: str,
        num_classes: int = 4,
        sequence_length: int = 100,
        embedding_dim: int = 300,
        max_features: int = 10000,
        model_dir: str = 'models'
    ):
        """
        Initialize DNN model manager
        
        Args:
            model_name: Name of the model (TextCNN, GRU, LSTM)
            num_classes: Number of classification labels
            sequence_length: Maximum sequence length for text input
            embedding_dim: Dimension of word embeddings
            max_features: Maximum number of words in vocabulary
            model_dir: Directory to save models
        """
        self.model_name = model_name
        self.num_classes = num_classes
        self.sequence_length = sequence_length
        self.embedding_dim = embedding_dim
        self.max_features = max_features
        self.model_dir = os.path.join(model_dir, model_name.lower())
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        
        # Create model directory if it doesn't exist
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Initialize tokenizer and model
        self.tokenizer = None
        self.model = None
        
        # Validate model name
        if model_name not in self.MODEL_BUILDERS:
            self.logger.error(f"Model {model_name} not found in available configurations")
            raise ValueError(f"Model {model_name} not supported. Available models: {list(self.MODEL_BUILDERS.keys())}")
    
    def create_text_cnn_model(
        self, 
        num_words: int,
        sequence_length: Optional[int] = None,
        embedding_dim: Optional[int] = None
    ) -> Model:
        """
        Create Text CNN model for text classification
        """
        if sequence_length is None:
            sequence_length = self.sequence_length
            
        if embedding_dim is None:
            embedding_dim = self.embedding_dim
        
        inputs = Input(shape=(sequence_length,), dtype='int32')

        embedding = Embedding(
            input_dim=num_words,
            output_dim=embedding_dim,
            input_length=sequence_length
        )(inputs)

        reshape = Reshape((sequence_length, embedding_dim, 1))(embedding)

        conv_blocks = []
        filter_sizes = [2, 3, 4]
        num_filters = 64

        for filter_size in filter_sizes:
            conv = Conv2D(
                num_filters,
                kernel_size=(filter_size, embedding_dim),
                activation='relu',
                padding='valid'
            )(reshape)

            pool = MaxPool2D(
                pool_size=(sequence_length - filter_size + 1, 1)
            )(conv)

            conv_blocks.append(pool)

        z = Concatenate(axis=1)(conv_blocks)
        z = Flatten()(z)
        z = Dropout(0.5)(z)

        outputs = Dense(self.num_classes, activation='softmax')(z)

        model = Model(inputs=inputs, outputs=outputs)
        return model
    
    def create_gru_model(
        self, 
        num_words: int, 
        sequence_length: Optional[int] = None,
        embedding_dim: Optional[int] = None
    ) -> Model:
        """
        Create GRU model for text classification
        """
        if sequence_length is None:
            sequence_length = self.sequence_length
            
        if embedding_dim is None:
            embedding_dim = self.embedding_dim
        
        inputs = Input(shape=(sequence_length,))

        x = Embedding(
            input_dim=num_words,
            output_dim=embedding_dim,
            input_length=sequence_length
        )(inputs)

        x = SpatialDropout1D(0.2)(x)
        x = Bidirectional(GRU(100, return_sequences=True))(x)

        avg_pool = GlobalAveragePooling1D()(x)
        max_pool = GlobalMaxPooling1D()(x)

        x = concatenate([avg_pool, max_pool])
        x = Dropout(0.5)(x)

        outputs = Dense(self.num_classes, activation='softmax')(x)

        model = Model(inputs=inputs, outputs=outputs)
        return model
    
    def create_lstm_model(
        self, 
        num_words: int, 
        sequence_length: Optional[int] = None,
        embedding_dim: Optional[int] = None
    ) -> Model:
        """
        Create LSTM model for text classification
        """
        if sequence_length is None:
            sequence_length = self.sequence_length
            
        if embedding_dim is None:
            embedding_dim = self.embedding_dim
        
        inputs = Input(shape=(sequence_length,))

        x = Embedding(
            input_dim=num_words,
            output_dim=embedding_dim,
            input_length=sequence_length
        )(inputs)

        x = SpatialDropout1D(0.2)(x)
        x = Bidirectional(LSTM(100, return_sequences=True))(x)

        avg_pool = GlobalAveragePooling1D()(x)
        max_pool = GlobalMaxPooling1D()(x)

        x = concatenate([avg_pool, max_pool])
        x = Dropout(0.5)(x)

        outputs = Dense(self.num_classes, activation='softmax')(x)

        model = Model(inputs=inputs, outputs=outputs)
        return model
    
    def _get_model_builder(self) -> Callable:
        """Get model builder function based on model name"""
        builder_name = self.MODEL_BUILDERS[self.model_name]
        builder_func = getattr(self, builder_name)
        return builder_func
    
    def prepare_tokenizer(self, texts: List[str]) -> Tokenizer:
        """
        Prepare tokenizer for text data
        """
        self.logger.info(f"Preparing tokenizer with max_features={self.max_features}")
        self.tokenizer = Tokenizer(
            num_words=self.max_features,
            lower=True,
            oov_token='<OOV>'
        )
        self.tokenizer.fit_on_texts(texts)
        return self.tokenizer
    
    def prepare_embedding_matrix(self) -> Tuple[np.ndarray, int]:
        """
        Prepare embedding matrix for the model
        """
        if self.tokenizer is None:
            self.logger.error("Tokenizer not initialized. Call prepare_tokenizer() first")
            raise ValueError("Tokenizer not initialized")
        
        word_index = self.tokenizer.word_index
        num_words = min(len(word_index) + 1, self.max_features)
        
        # Initialize with random embeddings (could be replaced with pre-trained)
        embedding_matrix = np.random.random((num_words, self.embedding_dim))
        embedding_matrix[0] = 0  # Reserve 0 for padding
        
        return embedding_matrix, num_words
    
    def build_model(self) -> Model:
        """
        Build the model based on model name
        """
        if self.tokenizer is None:
            self.logger.error("Tokenizer not initialized. Call prepare_tokenizer() first")
            raise ValueError("Tokenizer not initialized")
        
        # Prepare embedding matrix
        embedding_matrix, num_words = self.prepare_embedding_matrix()
        
        # Get model builder function
        model_builder = self._get_model_builder()
        
        # Build model
        self.logger.info(f"Building {self.model_name} model")
        self.model = model_builder(
            num_words=num_words,
            sequence_length=self.sequence_length,
            embedding_dim=self.embedding_dim
        )
        
        # Compile model
        self.compile_model()
        
        return self.model
    
    def compile_model(
        self,
        learning_rate: float = 0.001,
        loss: str = 'categorical_crossentropy',
        metrics: List[str] = ['accuracy']
    ) -> None:
        """
        Compile the model with specified parameters
        """
        if self.model is None:
            self.logger.error("Model not initialized")
            raise ValueError("Model not initialized")
        
        self.model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss=loss,
            metrics=metrics
        )
        
        self.logger.info(f"Model compiled with learning_rate={learning_rate}, loss={loss}")
    
    def train(
        self,
        train_data: np.ndarray,
        train_labels: np.ndarray,
        validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        batch_size: int = 64,
        epochs: int = 20,
        callbacks: Optional[List] = None,
        verbose: int = 1
    ) -> Dict[str, Any]:
        """
        Train the model on the provided data
        """
        if self.model is None:
            self.logger.error("Model not initialized. Call build_model() first")
            raise ValueError("Model not initialized")
        
        # Setup default callbacks if not provided
        if callbacks is None:
            callbacks = [
                EarlyStopping(
                    monitor='val_loss',
                    patience=3,
                    restore_best_weights=True
                ),
                ModelCheckpoint(
                    filepath=os.path.join(self.model_dir, 'best_model.h5'),
                    monitor='val_loss',
                    save_best_only=True
                ),
                CSVLogger(os.path.join(self.model_dir, 'training_log.csv')),
                TensorBoard(log_dir=os.path.join(self.model_dir, 'logs'))
            ]
        
        # Start training
        self.logger.info(f"Starting {self.model_name} training...")
        start_time = time.time()
        
        history = self.model.fit(
            train_data,
            train_labels,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )
        
        training_time = time.time() - start_time
        self.logger.info(f"{self.model_name} training completed in {training_time:.2f} seconds")
        
        # Save model
        self.save_model()
        
        return {
            'history': history,
            'training_time': training_time
        }
    
    def predict(self, test_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
        """
        Make predictions on test data
        """
        if self.model is None:
            self.logger.error("Model not initialized or trained")
            raise ValueError("Model not initialized or trained")
        
        self.logger.info(f"Starting {self.model_name} prediction...")
        start_time = time.time()
        
        # Get predictions
        y_pred_proba = self.model.predict(test_data)
        y_pred_classes = np.argmax(y_pred_proba, axis=1)
        
        inference_time = time.time() - start_time
        avg_inference_time = inference_time / len(test_data)
        
        self.logger.info(f"{self.model_name} prediction completed in {inference_time:.2f} seconds")
        self.logger.info(f"Average inference time per sample: {avg_inference_time:.4f} seconds")
        
        metrics = {
            'inference_time': inference_time,
            'avg_inference_time': avg_inference_time
        }
        
        return y_pred_proba, y_pred_classes, metrics
    
    def predict_single(self, text: str) -> int:
        """
        Make a prediction on a single text input
        """
        if self.model is None or self.tokenizer is None:
            self.logger.error("Model or tokenizer not initialized")
            raise ValueError("Model or tokenizer not initialized")
        
        # Convert text to sequence
        sequence = self.tokenizer.texts_to_sequences([text])
        
        # Pad sequence
        padded_sequence = pad_sequences(
            sequence,
            maxlen=self.sequence_length,
            padding='post',
            truncating='post'
        )
        
        # Make prediction
        prediction = self.model.predict(padded_sequence)
        predicted_class = np.argmax(prediction, axis=1)[0]
        
        return predicted_class
    
    def save_model(self, filepath: Optional[str] = None):
        """
        Save the model and tokenizer
        """
        if self.model is None:
            self.logger.error("Model not initialized")
            raise ValueError("Model not initialized")
        
        if filepath is None:
            filepath = os.path.join(self.model_dir, 'model.h5')
        
        # Save model
        self.model.save(filepath)
        self.logger.info(f"Model saved to {filepath}")
        
        # Save tokenizer
        if self.tokenizer is not None:
            import pickle
            tokenizer_path = os.path.join(self.model_dir, 'tokenizer.pickle')
            with open(tokenizer_path, 'wb') as handle:
                pickle.dump(self.tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
            self.logger.info(f"Tokenizer saved to {tokenizer_path}")
    
    def load_model(self, model_path: Optional[str] = None, tokenizer_path: Optional[str] = None):
        """
        Load a saved model and tokenizer
        """
        # Set default paths if not provided
        if model_path is None:
            model_path = os.path.join(self.model_dir, 'model.h5')
        
        if tokenizer_path is None:
            tokenizer_path = os.path.join(self.model_dir, 'tokenizer.pickle')
        
        # Load model
        try:
            self.model = load_model(model_path)
            self.logger.info(f"Model loaded from {model_path}")
        except Exception as e:
            self.logger.error(f"Error loading model from {model_path}: {str(e)}")
            raise e
        
        # Load tokenizer
        try:
            import pickle
            with open(tokenizer_path, 'rb') as handle:
                self.tokenizer = pickle.load(handle)
            self.logger.info(f"Tokenizer loaded from {tokenizer_path}")
        except Exception as e:
            self.logger.error(f"Error loading tokenizer from {tokenizer_path}: {str(e)}")
            raise e