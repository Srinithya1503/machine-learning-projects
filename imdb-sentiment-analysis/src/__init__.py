"""
================================================================================
IMDb Sentiment Analysis - Package Initialization
================================================================================

This file makes the src/ directory a Python package and allows importing
modules from different locations.
"""

# Import main modules for easy access
from src.preprocessing import preprocess_text, clean_dataset, remove_html_tags, remove_urls
from src.eda import (
    generate_visualizations, 
    create_word_clouds,
    create_sentiment_distribution,
    analyze_sentiment_distribution
)
from src.modeling import (
    train_models,
    evaluate_models,
    save_models,
    load_model,
    load_vectorizer,
    predict_sentiment
)
from src.utils import (
    print_section,
    print_success,
    print_error,
    print_warning,
    create_output_directories,
    calculate_metrics,
    batch_predict
)

__version__ = "1.0.0"
__author__ = "Your Name"
__description__ = "IMDb Movie Reviews Sentiment Analysis"

# Package metadata
__all__ = [
    # Preprocessing
    'preprocess_text',
    'clean_dataset',
    'remove_html_tags',
    'remove_urls',
    
    # EDA
    'generate_visualizations',
    'create_word_clouds',
    'create_sentiment_distribution',
    'analyze_sentiment_distribution',
    
    # Modeling
    'train_models',
    'evaluate_models',
    'save_models',
    'load_model',
    'load_vectorizer',
    'predict_sentiment',
    
    # Utils
    'print_section',
    'print_success',
    'print_error',
    'print_warning',
    'create_output_directories',
    'calculate_metrics',
    'batch_predict'
]