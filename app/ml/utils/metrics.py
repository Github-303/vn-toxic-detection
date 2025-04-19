# app/ml/utils/metrics.py
"""
Metrics and evaluation utilities for ViHSD models
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any, Union

from sklearn.metrics import (
    f1_score, confusion_matrix, accuracy_score,
    precision_score, recall_score, classification_report
)

class ModelEvaluator:
    """
    Utility class for evaluating model performance
    """
    
    def __init__(self, labels: List[str] = None):
        """
        Initialize evaluator with label names
        
        Args:
            labels: List of label names for classification
        """
        self.labels = labels or ["Clean", "Offensive", "Hate", "Spam"]
    
    def compute_metrics(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute classification metrics
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
        
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        
        # F1 scores
        metrics['f1_micro'] = f1_score(y_true, y_pred, average='micro')
        metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro')
        metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted')
        
        # Class-wise F1 scores
        class_f1 = f1_score(y_true, y_pred, average=None)
        for i, label in enumerate(self.labels):
            if i < len(class_f1):
                metrics[f'f1_{label.lower()}'] = class_f1[i]
        
        # Precision scores
        metrics['precision_micro'] = precision_score(y_true, y_pred, average='micro')
        metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro')
        metrics['precision_weighted'] = precision_score(y_true, y_pred, average='weighted')
        
        # Recall scores
        metrics['recall_micro'] = recall_score(y_true, y_pred, average='micro')
        metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro')
        metrics['recall_weighted'] = recall_score(y_true, y_pred, average='weighted')
        
        return metrics
    
    def get_confusion_matrix(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray
    ) -> np.ndarray:
        """
        Compute confusion matrix
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
        
        Returns:
            Confusion matrix
        """
        return confusion_matrix(y_true, y_pred)
    
    def get_classification_report(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray,
        output_dict: bool = False
    ) -> Union[str, Dict]:
        """
        Get classification report
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            output_dict: Whether to return report as a dictionary
        
        Returns:
            Classification report as string or dictionary
        """
        return classification_report(
            y_true, 
            y_pred, 
            target_names=self.labels,
            output_dict=output_dict
        )
    
    def plot_confusion_matrix(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray,
        figsize: Tuple[int, int] = (10, 8),
        cmap: str = "Blues",
        title: str = "Confusion Matrix",
        normalize: bool = False,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot confusion matrix
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            figsize: Figure size
            cmap: Colormap
            title: Plot title
            normalize: Whether to normalize confusion matrix
            save_path: Path to save the plot
        
        Returns:
            Matplotlib figure
        """
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Normalize if requested
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            title = f'Normalized {title}'
        
        # Create figure
        plt.figure(figsize=figsize)
        
        # Create DataFrame for Seaborn
        df_cm = pd.DataFrame(cm, index=self.labels, columns=self.labels)
        
        # Plot heatmap
        sns.heatmap(
            df_cm, 
            annot=True, 
            fmt='g' if not normalize else '.2f', 
            cmap=cmap,
            cbar=True,
            annot_kws={"size": 12}
        )
        
        plt.title(title, fontsize=16)
        plt.ylabel('True label', fontsize=12)
        plt.xlabel('Predicted label', fontsize=12)
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path)
        
        return plt.gcf()
    
    def plot_metrics_comparison(
        self,
        results: Dict[str, Dict[str, float]],
        metrics: List[str] = None,
        figsize: Tuple[int, int] = (12, 6),
        title: str = "Model Performance Comparison",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot comparison of metrics across models
        
        Args:
            results: Dictionary of model results
                {model_name: {metric_name: value}}
            metrics: List of metrics to compare
            figsize: Figure size
            title: Plot title
            save_path: Path to save the plot
        
        Returns:
            Matplotlib figure
        """
        if metrics is None:
            metrics = ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro']
        
        # Create figure
        plt.figure(figsize=figsize)
        
        model_names = list(results.keys())
        
        # Set width of bars
        bar_width = 0.8 / len(metrics)
        
        # Set positions of bars on X-axis
        index = np.arange(len(model_names))
        
        # Plot bars for each metric
        for i, metric in enumerate(metrics):
            values = [results[model].get(metric, 0) for model in model_names]
            plt.bar(
                index + i * bar_width, 
                values, 
                bar_width, 
                label=metric.replace('_', ' ').title()
            )
        
        # Customize plot
        plt.xlabel('Models', fontsize=12)
        plt.ylabel('Score', fontsize=12)
        plt.title(title, fontsize=16)
        plt.xticks(index + bar_width * (len(metrics) - 1) / 2, model_names)
        plt.ylim(0, 1.0)
        plt.legend()
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path)
        
        return plt.gcf()