"""
Air Quality ML Package - Restructured

This package has been restructured into two main components:

1. Pipeline: Core machine learning pipeline components
   - DataProcessor: Data loading and preprocessing
   - FeatureEngineer: Feature extraction and selection
   - ModelTrainer: Model training and comparison
   - Evaluator: Core model evaluation

2. Utils: Utilities and configuration
   - config: Configuration constants and settings
   - utils: General utility functions
   - evaluation_utils: Detailed evaluation functions with visualizations

Usage:
    from pipeline import DataProcessor, FeatureEngineer, ModelTrainer, Evaluator
    from utils.config import *
    from utils.evaluation_utils import evaluate_model_detailed
"""

__version__ = "1.0.0"
__author__ = "NLP Pipeline ML Workshop - Restructured"

# Import main pipeline classes for backward compatibility
from .pipeline.data_processor import DataProcessor
from .pipeline.feature_engineer import FeatureEngineer
from .pipeline.model_trainer import ModelTrainer
from .pipeline.evaluator import Evaluator

__all__ = [
    'DataProcessor',
    'FeatureEngineer', 
    'ModelTrainer',
    'Evaluator'
]
