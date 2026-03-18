"""
================================================================================
Model Training & Evaluation Module
================================================================================

Functions for training, evaluating, and saving sentiment classification models.

Includes:
- Model training (Logistic Regression, Naïve Bayes, SVM)
- Model evaluation (Accuracy, Precision, Recall, F1-Score)
- Confusion matrix generation
- Model persistence (save/load)
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    confusion_matrix,
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================================
# MODEL TRAINING
# ============================================================================

def train_models(X_train, y_train, models_dir='models/'):
    """
    Train three classification models.
    
    Args:
        X_train (array-like): Training features
        y_train (array-like): Training labels
        models_dir (str): Directory to save models
    
    Returns:
        dict: Dictionary of trained models
        TfidfVectorizer: Fitted vectorizer
    """
    
    print("\n" + "="*60)
    print("TRAINING SENTIMENT CLASSIFICATION MODELS")
    print("="*60)
    
    # Create output directory
    os.makedirs(models_dir, exist_ok=True)
    
    # Vectorize text
    print("\n1️⃣  Vectorizing text data...")
    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words='english',
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    print(f"   ✓ Vectorization complete")
    print(f"   • Features: {X_train_vec.shape[1]}")
    print(f"   • Samples: {X_train_vec.shape[0]}")
    
    # Initialize models
    models = {}
    
    # Model 1: Logistic Regression
    print("\n2️⃣  Training Logistic Regression...")
    lr_model = LogisticRegression(
        max_iter=1000,
        random_state=42,
        n_jobs=-1,
        verbose=0
    )
    lr_model.fit(X_train_vec, y_train)
    models['Logistic Regression'] = lr_model
    print("   ✓ Training complete")
    
    # Model 2: Multinomial Naïve Bayes
    print("3️⃣  Training Multinomial Naïve Bayes...")
    nb_model = MultinomialNB()
    nb_model.fit(X_train_vec, y_train)
    models['Naive Bayes'] = nb_model
    print("   ✓ Training complete")
    
    # Model 3: Linear SVM
    print("4️⃣  Training Linear SVM...")
    svm_model = LinearSVC(
        max_iter=2000,
        random_state=42,
        verbose=0
    )
    svm_model.fit(X_train_vec, y_train)
    models['SVM'] = svm_model
    print("   ✓ Training complete")
    
    print("\n" + "="*60)
    print(f"✓ All {len(models)} models trained successfully")
    print("="*60)
    
    return models, vectorizer

# ============================================================================
# MODEL EVALUATION
# ============================================================================

def evaluate_models(models, vectorizer, X_test, y_test, output_dir='outputs/'):
    """
    Evaluate all trained models on test set.
    
    Args:
        models (dict): Dictionary of trained models
        vectorizer (TfidfVectorizer): Fitted vectorizer
        X_test (array-like): Test features (raw text)
        y_test (array-like): Test labels
        output_dir (str): Directory to save results
    
    Returns:
        pd.DataFrame: Model performance metrics
    """
    
    print("\n" + "="*60)
    print("MODEL EVALUATION")
    print("="*60)
    
    # Vectorize test data
    X_test_vec = vectorizer.transform(X_test)
    
    results = []
    
    for model_name, model in models.items():
        print(f"\n{model_name}:")
        print("-" * 40)
        
        # Make predictions
        y_pred = model.predict(X_test_vec)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, pos_label='positive')
        recall = recall_score(y_test, y_pred, pos_label='positive')
        f1 = f1_score(y_test, y_pred, pos_label='positive')
        
        # Get confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Store results
        results.append({
            'Model': model_name,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'TP': cm[1, 1],
            'TN': cm[0, 0],
            'FP': cm[0, 1],
            'FN': cm[1, 0]
        })
        
        # Print metrics
        print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        
        # Print classification report
        print(f"\n  Classification Report:")
        print(classification_report(y_test, y_pred, zero_division=0))
        
        # Generate confusion matrix visualization
        generate_confusion_matrix(y_test, y_pred, model_name, output_dir)
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    print("\n" + "="*60)
    print("SUMMARY TABLE")
    print("="*60)
    print(results_df[['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score']].to_string(index=False))
    print("="*60)
    
    return results_df

def generate_confusion_matrix(y_test, y_pred, model_name, output_dir='outputs/'):
    """
    Generate and save confusion matrix visualization.
    
    Args:
        y_test (array-like): True labels
        y_pred (array-like): Predicted labels
        model_name (str): Name of model
        output_dir (str): Directory to save image
    """
    
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['Negative', 'Positive'],
        yticklabels=['Negative', 'Positive'],
        cbar_kws={'label': 'Count'}
    )
    plt.title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    # Save
    os.makedirs(os.path.join(output_dir, 'visualizations'), exist_ok=True)
    filename = model_name.lower().replace(' ', '_') + '_confusion_matrix.png'
    filepath = os.path.join(output_dir, 'visualizations', filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()

# ============================================================================
# MODEL SAVING & LOADING
# ============================================================================

def save_models(models, models_dir='models/'):
    """
    Save trained models to pickle files.
    
    Args:
        models (dict): Dictionary of trained models
        models_dir (str): Directory to save models
    """
    
    print("\n" + "="*60)
    print("SAVING MODELS")
    print("="*60)
    
    os.makedirs(models_dir, exist_ok=True)
    
    for model_name, model in models.items():
        # Create filename
        filename = model_name.lower().replace(' ', '_') + '.pkl'
        filepath = os.path.join(models_dir, filename)
        
        # Save model
        with open(filepath, 'wb') as f:
            pickle.dump(model, f)
        
        # Get file size
        file_size = os.path.getsize(filepath) / 1024  # KB
        print(f"✓ {model_name:20} → {filename:30} ({file_size:.1f} KB)")
    
    print("="*60)

def load_model(model_name, models_dir='models/'):
    """
    Load a trained model from pickle file.
    
    Args:
        model_name (str): Name of model to load
        models_dir (str): Directory containing models
    
    Returns:
        Model: Loaded model object
    
    Example:
        >>> model = load_model('Logistic Regression')
    """
    
    filename = model_name.lower().replace(' ', '_') + '.pkl'
    filepath = os.path.join(models_dir, filename)
    
    try:
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        print(f"✓ Loaded model: {model_name}")
        return model
    except FileNotFoundError:
        print(f"❌ Model not found: {filepath}")
        return None

def load_vectorizer(models_dir='models/'):
    """
    Load the TF-IDF vectorizer.
    
    Args:
        models_dir (str): Directory containing models
    
    Returns:
        TfidfVectorizer: Loaded vectorizer
    """
    
    filepath = os.path.join(models_dir, 'tfidf_vectorizer.pkl')
    
    try:
        with open(filepath, 'rb') as f:
            vectorizer = pickle.load(f)
        print("✓ Loaded vectorizer")
        return vectorizer
    except FileNotFoundError:
        print(f"❌ Vectorizer not found: {filepath}")
        return None

# ============================================================================
# PREDICTION FUNCTION
# ============================================================================

def predict_sentiment(review_text, model, vectorizer):
    """
    Predict sentiment for a single review.
    
    Args:
        review_text (str): Raw review text
        model: Trained sentiment model
        vectorizer (TfidfVectorizer): Fitted vectorizer
    
    Returns:
        tuple: (prediction, confidence)
    
    Example:
        >>> pred, conf = predict_sentiment(review, model, vectorizer)
        >>> print(f"{pred} ({conf*100:.1f}%)")
    """
    
    # Preprocess review
    from src.preprocessing import preprocess_text
    cleaned = preprocess_text(review_text)
    
    # Vectorize
    X = vectorizer.transform([cleaned])
    
    # Predict
    prediction = model.predict(X)[0]
    
    # Get confidence
    if hasattr(model, 'predict_proba'):
        confidence = model.predict_proba(X).max()
    else:
        confidence = abs(model.decision_function(X)[0])
    
    return prediction, confidence