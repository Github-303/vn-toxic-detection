# app/ml/data/dataset.py
"""
Dataset handling utilities for ViHSD models
"""
import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union

class BuildDataset(torch.utils.data.Dataset):
    """Dataset builder for transformer models"""
    def __init__(self, encodings: Dict[str, List], labels: np.ndarray):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self) -> int:
        return len(self.labels)


class DataLoader:
    """Data loading and preparation for ViHSD models"""
    
    @staticmethod
    def load_data(
        train_path: str, 
        dev_path: str, 
        test_path: str,
        text_column: str = 'free_text',
        label_column: str = 'label_id',
        verbose: bool = True
    ) -> Tuple[
        np.ndarray, np.ndarray, 
        np.ndarray, np.ndarray, 
        np.ndarray, np.ndarray
    ]:
        """Load and prepare raw data"""
        
        # Load datasets
        train_data = pd.read_csv(train_path)
        dev_data = pd.read_csv(dev_path)
        test_data = pd.read_csv(test_path)
        
        # Convert text data to string and handle NaN values
        X_train = train_data[text_column].fillna('').astype(str).values
        X_dev = dev_data[text_column].fillna('').astype(str).values
        X_test = test_data[text_column].fillna('').astype(str).values
        
        # Get labels
        y_train = train_data[label_column].values
        y_dev = dev_data[label_column].values
        y_test = test_data[label_column].values
        
        if verbose:
            # Print dataset statistics
            print(f"\n=== DATASET STATISTICS ===")
            print(f"Train data size: {len(train_data)}")
            print(f"Dev data size: {len(dev_data)}")
            print(f"Test data size: {len(test_data)}")
            
            # Print label distribution
            label_names = ['Clean', 'Offensive', 'Hate', 'Spam']
            print("\n=== DATASET LABEL DISTRIBUTION ===")
            
            for dataset_name, labels in [
                ("Train", y_train),
                ("Dev", y_dev),
                ("Test", y_test)
            ]:
                unique, counts = np.unique(labels, return_counts=True)
                distribution = dict(zip(unique, counts))
                print(f"\n{dataset_name} Dataset:")
                for label, count in distribution.items():
                    if label < len(label_names):
                        label_name = label_names[label]
                        percentage = count / len(labels) * 100
                        print(f"{label_name}: {count} ({percentage:.2f}%)")
        
        return X_train, y_train, X_dev, y_dev, X_test, y_test
    
    @staticmethod
    def prepare_transformer_data(
        X_train: np.ndarray, 
        X_dev: np.ndarray, 
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_dev: np.ndarray,
        y_test: np.ndarray,
        tokenizer: Any,
        max_length: int = 100
    ) -> Tuple[BuildDataset, BuildDataset, BuildDataset]:
        """Prepare data for transformer models"""
        
        # Tokenize and encode data
        train_encodings = tokenizer(
            X_train.tolist(),
            truncation=True,
            padding=True,
            max_length=max_length
        )
        
        dev_encodings = tokenizer(
            X_dev.tolist(),
            truncation=True,
            padding=True,
            max_length=max_length
        )
        
        test_encodings = tokenizer(
            X_test.tolist(),
            truncation=True,
            padding=True,
            max_length=max_length
        )
        
        # Create PyTorch datasets
        train_dataset = BuildDataset(train_encodings, y_train)
        dev_dataset = BuildDataset(dev_encodings, y_dev)
        test_dataset = BuildDataset(test_encodings, y_test)
        
        return train_dataset, dev_dataset, test_dataset
    
    @staticmethod
    def prepare_dnn_data(
        X_train: List[str],
        X_dev: List[str],
        X_test: List[str],
        y_train: np.ndarray,
        y_dev: np.ndarray,
        y_test: np.ndarray,
        tokenizer: Any,
        sequence_length: int = 100,
        num_classes: int = 4
    ) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray, 
        np.ndarray, np.ndarray, np.ndarray
    ]:
        """Prepare data for DNN models"""
        from tensorflow.keras.utils import to_categorical
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        
        # Convert texts to sequences
        train_sequences = tokenizer.texts_to_sequences(X_train)
        dev_sequences = tokenizer.texts_to_sequences(X_dev)
        test_sequences = tokenizer.texts_to_sequences(X_test)
        
        # Pad sequences
        train_pad = pad_sequences(
            train_sequences, 
            maxlen=sequence_length, 
            padding='post', 
            truncating='post'
        )
        
        dev_pad = pad_sequences(
            dev_sequences, 
            maxlen=sequence_length, 
            padding='post', 
            truncating='post'
        )
        
        test_pad = pad_sequences(
            test_sequences, 
            maxlen=sequence_length, 
            padding='post', 
            truncating='post'
        )
        
        # Convert labels to categorical
        train_labels_cat = to_categorical(y_train, num_classes=num_classes)
        dev_labels_cat = to_categorical(y_dev, num_classes=num_classes)
        
        return train_pad, dev_pad, test_pad, train_labels_cat, dev_labels_cat, y_test