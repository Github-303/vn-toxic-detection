# app/ml/data/preprocessing.py
"""
Data preprocessing utilities for ViHSD models
"""
import re
import string
import logging
from typing import List, Tuple, Optional, Set, Union, Dict
import numpy as np
import pandas as pd
from underthesea import word_tokenize
from pathlib import Path

logger = logging.getLogger(__name__)

# Handle VnCoreNLP import conditionally
try:
    from vncorenlp import VnCoreNLP
    VNCORENLP_AVAILABLE = True
except ImportError:
    VNCORENLP_AVAILABLE = False

class TextPreprocessor:
    """
    Text preprocessing for Vietnamese text classification
    """
    def __init__(
        self,
        max_length: int = 256,
        vocab_size: int = 20000
    ):
        """
        Initialize text preprocessor
        
        Args:
            max_length: Maximum sequence length
            vocab_size: Maximum vocabulary size for DNN models
        """
        self.max_length = max_length
        self.vocab_size = vocab_size
        
        # Load stopwords if available
        self.stopwords = set()
        try:
            stopwords_path = Path(__file__).parent / "vietnamese-stopwords.txt"
            with open(stopwords_path, "r", encoding="utf-8") as f:
                self.stopwords = set(f.read().splitlines())
        except Exception as e:
            logger.warning(f"Could not load stopwords: {str(e)}")
        
        # Initialize VnCoreNLP if available
        self.vncorenlp = None
        if VNCORENLP_AVAILABLE:
            try:
                self.vncorenlp = VnCoreNLP(
                    annotators="wseg", 
                    max_heap_size='-Xmx500m'
                )
            except Exception as e:
                print(f"Error initializing VnCoreNLP: {e}")
                self.vncorenlp = None
    
    def _load_stopwords(self, stopwords_path: str) -> Set[str]:
        """Load Vietnamese stopwords from file"""
        stopwords = set()
        try:
            with open(stopwords_path, "r", encoding='utf-8') as f:
                for line in f:
                    word = line.strip('\n')
                    stopwords.add(word)
        except Exception as e:
            print(f"Error loading stopwords: {e}")
        return stopwords
    
    def filter_stop_words(self, text: str) -> str:
        """Remove stopwords from text"""
        if not self.stopwords:
            return text
        
        words = text.split()
        filtered_words = [word for word in words if word not in self.stopwords]
        return ' '.join(filtered_words)
    
    def remove_emojis(self, text: str) -> str:
        """Remove emojis from text"""
        emoji_pattern = re.compile(
            pattern="["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
            "]+", 
            flags=re.UNICODE
        )
        return emoji_pattern.sub(r'', text)
    
    def tokenize(self, text: str) -> str:
        """Tokenize text using VnCoreNLP if available"""
        if self.vncorenlp is None:
            return text
        
        try:
            tokenized_text = ""
            sentences = self.vncorenlp.tokenize(text)
            for sentence in sentences:
                tokenized_text += " ".join(sentence)
            return tokenized_text
        except Exception as e:
            print(f"Error tokenizing text: {e}")
            return text
    
    def normalize_urls(self, text: str) -> Tuple[str, int]:
        """Normalize URLs and count them"""
        url_pattern = r"http\S+|www\S+|https\S+"
        urls = re.findall(url_pattern, text, flags=re.MULTILINE)
        url_count = len(urls)
        
        # Replace URLs with placeholder
        text = re.sub(url_pattern, " <URL> ", text, flags=re.MULTILINE)
        return text, url_count
    
    def normalize_emails(self, text: str) -> Tuple[str, int]:
        """Normalize email addresses and count them"""
        email_pattern = r"\S+@\S+"
        emails = re.findall(email_pattern, text)
        email_count = len(emails)
        
        # Replace emails with placeholder
        text = re.sub(email_pattern, " <EMAIL> ", text)
        return text, email_count
    
    def normalize_phone_numbers(self, text: str) -> Tuple[str, int]:
        """Normalize phone numbers and count them"""
        phone_pattern = r"[\+]?[(]?[0-9]{3}[)]?[-\s\.]?[0-9]{3}[-\s\.]?[0-9]{4,6}"
        phones = re.findall(phone_pattern, text)
        phone_count = len(phones)
        
        # Replace phone numbers with placeholder
        text = re.sub(phone_pattern, " <PHONE> ", text)
        return text, phone_count
    
    def clean_text(self, text: str) -> Tuple[str, Dict[str, int]]:
        """
        Clean text while preserving spam detection features
        
        Args:
            text: Input text
            
        Returns:
            Tuple of (cleaned_text, feature_counts)
        """
        # Convert to lowercase
        text = text.lower()
        
        # Count and normalize special patterns
        text, url_count = self.normalize_urls(text)
        text, email_count = self.normalize_emails(text)
        text, phone_count = self.normalize_phone_numbers(text)
        
        # Count special characters
        special_char_count = len(re.findall(r'[^\w\s]', text))
        
        # Count repeated characters (e.g., "hellooooo")
        repeated_char_count = len(re.findall(r'(.)\1{2,}', text))
        
        # Count numbers
        number_count = len(re.findall(r'\d+', text))
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        feature_counts = {
            'urls': url_count,
            'emails': email_count,
            'phones': phone_count,
            'special_chars': special_char_count,
            'repeated_chars': repeated_char_count,
            'numbers': number_count
        }
        
        return text, feature_counts
    
    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """Remove stopwords from token list"""
        return [token for token in tokens if token not in self.stopwords]
    
    def preprocess(
        self, 
        text: str, 
        tokenized: bool = True, 
        lowercased: bool = True,
        remove_emoji: bool = True,
        remove_stopwords: bool = True,
        return_features: bool = False
    ) -> Union[str, Tuple[str, Dict[str, int]]]:
        """Full preprocessing pipeline"""
        if pd.isna(text) or text is None:
            return ("", {}) if return_features else ""
        
        text = str(text)
        
        # Clean text and get feature counts
        text, feature_counts = self.clean_text(text)
        
        # Apply preprocessing steps
        if remove_stopwords:
            text = self.filter_stop_words(text)
        
        if remove_emoji:
            text = self.remove_emojis(text)
        
        if tokenized and self.vncorenlp is not None:
            text = self.tokenize(text)
        
        if return_features:
            return text, feature_counts
        return text
    
    def process_features(
        self, 
        texts: List[str], 
        labels: Optional[np.ndarray] = None, 
        **kwargs
    ) -> Union[List[str], Tuple[List[str], np.ndarray]]:
        """Process a list of text features and optionally their labels"""
        processed_texts = [self.preprocess(text, **kwargs) for text in texts]
        
        # If labels are provided, remove empty texts and corresponding labels
        if labels is not None:
            filtered_texts = []
            filtered_labels = []
            
            for i, text in enumerate(processed_texts):
                if text.strip():  # Keep non-empty texts
                    filtered_texts.append(text)
                    filtered_labels.append(labels[i])
            
            return filtered_texts, np.array(filtered_labels)
        
        return processed_texts

    def text_to_sequence(self, text: str) -> np.ndarray:
        """
        Convert text to sequence for DNN models
        
        Args:
            text: Input text
            
        Returns:
            Numpy array of token indices
        """
        # This is a placeholder - implement actual sequence conversion
        # based on your vocabulary and tokenization strategy
        sequence = np.zeros(self.max_length)
        tokens = self.tokenize(text)[:self.max_length]
        
        for i, token in enumerate(tokens):
            # Convert token to index based on your vocabulary
            # This is just a placeholder hash function
            sequence[i] = hash(token) % self.vocab_size
        
        return sequence

class MinimalTextCleaner:
    """
    Minimal text cleaning for DNN models
    """
    def __init__(self, max_length: int = 1000):
        self.max_length = max_length
    
    def clean(self, text: str) -> str:
        """Minimal text cleaning for raw data"""
        if pd.isna(text) or text is None:
            return ""

        text = str(text)
        text = re.sub(r'\s+', ' ', text).strip()

        if len(text) > self.max_length:
            text = text[:self.max_length]

        return text