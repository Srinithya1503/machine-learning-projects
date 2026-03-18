"""
================================================================================
Utility Functions Module
================================================================================

Helper functions for:
- Directory creation
- Console output formatting
- Metrics calculation
- File operations
"""

import os
import sys
from pathlib import Path
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report

# ============================================================================
# CONSOLE OUTPUT FUNCTIONS
# ============================================================================

def print_section(title):
    """
    Print a formatted section header.
    
    Args:
        title (str): Section title to display
    
    Example:
        >>> print_section("Step 1: Loading Data")
    """
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80)

def print_success(message):
    """
    Print a success message with checkmark.
    
    Args:
        message (str): Success message
    """
    print(f"  ✓ {message}")

def print_error(message):
    """
    Print an error message with X mark.
    
    Args:
        message (str): Error message
    """
    print(f"  ❌ {message}")

def print_warning(message):
    """
    Print a warning message.
    
    Args:
        message (str): Warning message
    """
    print(f"  ⚠️  {message}")

def print_info(message):
    """
    Print an info message.
    
    Args:
        message (str): Info message
    """
    print(f"  ℹ️  {message}")

# ============================================================================
# DIRECTORY MANAGEMENT
# ============================================================================

def create_output_directories(directories):
    """
    Create multiple output directories if they don't exist.
    
    Args:
        directories (list): List of directory paths to create
    
    Example:
        >>> create_output_directories(['models/', 'outputs/', 'data/processed/'])
    """
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)

def get_project_structure():
    """
    Get the project directory structure.
    
    Returns:
        dict: Dictionary of project paths
    """
    
    project_root = Path(__file__).parent.parent
    
    structure = {
        'root': str(project_root),
        'data_raw': str(project_root / 'data' / 'raw'),
        'data_processed': str(project_root / 'data' / 'processed'),
        'notebooks': str(project_root / 'notebooks'),
        'src': str(project_root / 'src'),
        'models': str(project_root / 'models'),
        'outputs': str(project_root / 'outputs'),
        'visualizations': str(project_root / 'outputs' / 'visualizations'),
        'results': str(project_root / 'outputs' / 'results')
    }
    
    return structure

# ============================================================================
# FILE OPERATIONS
# ============================================================================

def save_dataframe_to_csv(df, filepath, index=False):
    """
    Save DataFrame to CSV file.
    
    Args:
        df (pd.DataFrame): DataFrame to save
        filepath (str): Output file path
        index (bool): Whether to save index
    
    Returns:
        bool: True if successful, False otherwise
    """
    
    try:
        # Create parent directory if needed
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        df.to_csv(filepath, index=index)
        print_success(f"Saved to: {filepath}")
        return True
    
    except Exception as e:
        print_error(f"Failed to save file: {e}")
        return False

def load_dataframe_from_csv(filepath):
    """
    Load DataFrame from CSV file.
    
    Args:
        filepath (str): Path to CSV file
    
    Returns:
        pd.DataFrame: Loaded DataFrame or None if error
    """
    
    try:
        df = pd.read_csv(filepath)
        print_success(f"Loaded: {filepath}")
        return df
    
    except FileNotFoundError:
        print_error(f"File not found: {filepath}")
        return None
    
    except Exception as e:
        print_error(f"Error loading file: {e}")
        return None

# ============================================================================
# METRICS & EVALUATION
# ============================================================================

def calculate_metrics(y_true, y_pred):
    """
    Calculate classification metrics.
    
    Args:
        y_true (array-like): True labels
        y_pred (array-like): Predicted labels
    
    Returns:
        dict: Dictionary of metrics
    """
    
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, 
        f1_score, confusion_matrix
    )
    
    cm = confusion_matrix(y_true, y_pred)
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, pos_label='positive', zero_division=0),
        'recall': recall_score(y_true, y_pred, pos_label='positive', zero_division=0),
        'f1_score': f1_score(y_true, y_pred, pos_label='positive', zero_division=0),
        'true_positives': cm[1, 1],
        'true_negatives': cm[0, 0],
        'false_positives': cm[0, 1],
        'false_negatives': cm[1, 0]
    }
    
    return metrics

def print_metrics_summary(metrics):
    """
    Print a formatted summary of metrics.
    
    Args:
        metrics (dict): Dictionary of metrics from calculate_metrics()
    """
    
    print("\n" + "="*60)
    print("MODEL METRICS SUMMARY")
    print("="*60)
    print(f"Accuracy:      {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"Precision:     {metrics['precision']:.4f}")
    print(f"Recall:        {metrics['recall']:.4f}")
    print(f"F1-Score:      {metrics['f1_score']:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  True Positives:   {metrics['true_positives']}")
    print(f"  True Negatives:   {metrics['true_negatives']}")
    print(f"  False Positives:  {metrics['false_positives']}")
    print(f"  False Negatives:  {metrics['false_negatives']}")
    print("="*60)

# ============================================================================
# TEXT ANALYSIS UTILITIES
# ============================================================================

def get_text_length_stats(df, text_column='review'):
    """
    Get statistics about text lengths in a DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame with text column
        text_column (str): Name of text column
    
    Returns:
        dict: Dictionary of statistics
    """
    
    word_counts = df[text_column].apply(lambda x: len(str(x).split()))
    
    stats = {
        'min_words': word_counts.min(),
        'max_words': word_counts.max(),
        'mean_words': word_counts.mean(),
        'median_words': word_counts.median(),
        'std_words': word_counts.std(),
        'min_chars': df[text_column].apply(len).min(),
        'max_chars': df[text_column].apply(len).max(),
        'mean_chars': df[text_column].apply(len).mean(),
    }
    
    return stats

def print_text_statistics(stats):
    """
    Print formatted text statistics.
    
    Args:
        stats (dict): Dictionary from get_text_length_stats()
    """
    
    print("\nText Length Statistics:")
    print("-" * 40)
    print(f"Word Count:")
    print(f"  Min:    {stats['min_words']:.0f}")
    print(f"  Max:    {stats['max_words']:.0f}")
    print(f"  Mean:   {stats['mean_words']:.1f}")
    print(f"  Median: {stats['median_words']:.1f}")
    print(f"\nCharacter Count:")
    print(f"  Min:    {stats['min_chars']:.0f}")
    print(f"  Max:    {stats['max_chars']:.0f}")
    print(f"  Mean:   {stats['mean_chars']:.1f}")

# ============================================================================
# BATCH PREDICTION
# ============================================================================

def batch_predict(reviews, model, vectorizer):
    """
    Make predictions on multiple reviews.
    
    Args:
        reviews (list): List of review texts
        model: Trained sentiment model
        vectorizer: Fitted TF-IDF vectorizer
    
    Returns:
        pd.DataFrame: DataFrame with reviews and predictions
    """
    
    from src.preprocessing import preprocess_text
    
    predictions = []
    confidences = []
    
    for review in reviews:
        # Preprocess
        cleaned = preprocess_text(review)
        
        # Vectorize
        X = vectorizer.transform([cleaned])
        
        # Predict
        pred = model.predict(X)[0]
        
        # Confidence
        if hasattr(model, 'predict_proba'):
            conf = model.predict_proba(X).max()
        else:
            conf = abs(model.decision_function(X)[0])
        
        predictions.append(pred)
        confidences.append(conf)
    
    return pd.DataFrame({
        'review': reviews,
        'prediction': predictions,
        'confidence': confidences
    })

# ============================================================================
# VALIDATION UTILITIES
# ============================================================================

def validate_dataset(df, required_columns=None):
    """
    Validate dataset structure and content.
    
    Args:
        df (pd.DataFrame): Dataset to validate
        required_columns (list): List of required columns
    
    Returns:
        bool: True if valid, False otherwise
    """
    
    if required_columns is None:
        required_columns = ['review', 'sentiment']
    
    # Check if DataFrame is empty
    if df.empty:
        print_error("Dataset is empty")
        return False
    
    # Check required columns
    for col in required_columns:
        if col not in df.columns:
            print_error(f"Missing required column: {col}")
            return False
    
    # Check for null values in required columns
    null_counts = df[required_columns].isnull().sum()
    if null_counts.sum() > 0:
        print_warning("Null values found in dataset:")
        print(null_counts)
    
    print_success("Dataset validation passed")
    return True

# ============================================================================
# PROGRESS TRACKING
# ============================================================================

def print_progress_bar(current, total, prefix='', suffix='', length=50):
    """
    Print a progress bar to console.
    
    Args:
        current (int): Current iteration
        total (int): Total iterations
        prefix (str): Prefix text
        suffix (str): Suffix text
        length (int): Length of progress bar
    
    Example:
        >>> for i in range(100):
        ...     print_progress_bar(i+1, 100, prefix='Progress:')
    """
    
    percent = current / total
    filled = int(length * percent)
    bar = '█' * filled + '░' * (length - filled)
    
    sys.stdout.write(f'\r{prefix} |{bar}| {percent*100:.1f}% {suffix}')
    sys.stdout.flush()
    
    if current == total:
        print()  # New line at completion