"""
================================================================================
IMDb Movie Reviews Sentiment Analysis - Main Pipeline
================================================================================

A beginner-friendly end-to-end machine learning pipeline that:
1. Loads IMDb dataset (50K reviews)
2. Performs exploratory data analysis (EDA)
3. Cleans and preprocesses text
4. Trains 3 ML models (Logistic Regression, Naïve Bayes, SVM)
5. Evaluates and compares models
6. Saves trained models for production use
"""

import os
import sys
import pandas as pd
import numpy as np
import pickle
import warnings
from pathlib import Path
import io

# Ensure stdout and file writes use UTF-8 encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")


warnings.filterwarnings('ignore')

try:
    from src.preprocessing import preprocess_text, clean_dataset
    from src.eda import generate_visualizations, analyze_sentiment_distribution
    from src.modeling import train_models, evaluate_models, save_models
    from src.utils import create_output_directories, print_section, print_success, print_error
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Make sure you have the src/ folder with all necessary files.")
    sys.exit(1)

# ============================================================================
# CONFIGURATION
# ============================================================================

# File paths
RAW_DATA_PATH = "data/raw/IMDB Dataset.csv"
PROCESSED_DATA_PATH = "data/processed/cleaned_reviews.csv"
MODELS_DIR = "models/"
OUTPUTS_DIR = "outputs/"
VISUALIZATIONS_DIR = os.path.join(OUTPUTS_DIR, 'visualizations/')
RESULTS_DIR = os.path.join(OUTPUTS_DIR, 'results/')

# Hyperparameters
TEST_SIZE = 0.2
RANDOM_STATE = 42
MAX_FEATURES = 5000
SAMPLE_SIZE = None  # Set to integer to use smaller dataset (e.g., 5000)

# ============================================================================
# STEP 0: WELCOME MESSAGE & SETUP
# ============================================================================

def print_welcome():
    """Display welcome banner"""
    print("\n" + "="*80)
    print(" IMDb MOVIE REVIEWS SENTIMENT ANALYSIS  🎬")
    print("="*80)
    print("\nThis script will:")
    print("  1. Load IMDb dataset (50K reviews)")
    print("  2. Perform Exploratory Data Analysis (EDA)")
    print("  3. Clean and preprocess text data")
    print("  4. Train 3 machine learning models")
    print("  5. Evaluate and compare performance")
    print("  6. Save models for future predictions")
    print("\n" + "="*80 + "\n")

print_welcome()

# ============================================================================
# STEP 1: CREATE OUTPUT DIRECTORIES
# ============================================================================

def setup_directories():
    """Create necessary output directories"""
    print_section("Step 1: Setting Up Directories")
    
    directories = [MODELS_DIR, VISUALIZATIONS_DIR, RESULTS_DIR, 'data/processed']
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"  ✓ {directory}")
    
    print_success("All directories created/verified")

setup_directories()

# ============================================================================
# STEP 2: LOAD RAW DATA
# ============================================================================

def load_data():
    """Load dataset from CSV"""
    print_section("Step 2: Loading Data")
    
    try:
        df = pd.read_csv(RAW_DATA_PATH)
        print(f"  ✓ Successfully loaded: {RAW_DATA_PATH}")
        print(f"  • Total reviews: {len(df):,}")
        print(f"  • Columns: {list(df.columns)}")
        print(f"  • Shape: {df.shape}")
        print(f"  • Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        return df
    
    except FileNotFoundError:
        print_error(f"Dataset not found at: {RAW_DATA_PATH}")
        print("\n  To download the dataset:")
        print(" 1. Visit: https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews")
        print(" 2. Download imdb_dataset.csv")
        print(f"3. Place it in: data/raw/")
        sys.exit(1)
    
    except pd.errors.ParserError:
        print_error("Error parsing CSV file. Check file format.")
        sys.exit(1)

df = load_data()

# Use sample if specified (for testing)
if SAMPLE_SIZE and len(df) > SAMPLE_SIZE:
    print(f"\n Using sample of {SAMPLE_SIZE} reviews for testing")
    df = df.sample(n=SAMPLE_SIZE, random_state=RANDOM_STATE)

# ============================================================================
# STEP 3: DATA VALIDATION & CLEANING
# ============================================================================

def validate_data(data):
    """Check data quality"""
    print_section("Step 3: Data Validation")
    
    print(f"  • Rows: {len(data)}")
    print(f"  • Null values: {data.isnull().sum().sum()}")
    print(f"  • Duplicates: {data.duplicated().sum()}")
    
    # Check required columns
    required_cols = ['review', 'sentiment']
    for col in required_cols:
        if col not in data.columns:
            print_error(f"Missing required column: {col}")
            sys.exit(1)
    
    # Show first sample
    print(f"\n  Sample review:\n  {data['review'].iloc[0][:150]}...\n")
    print_success("Data validation passed")

validate_data(df)

# ============================================================================
# STEP 4: EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================================

def perform_eda(data):
    """Analyze dataset characteristics"""
    print_section("Step 4: Exploratory Data Analysis (EDA)")
    
    # Sentiment distribution
    print("\n Sentiment Distribution:")
    sentiment_counts = data['sentiment'].value_counts()
    for sentiment, count in sentiment_counts.items():
        percentage = (count / len(data)) * 100
        print(f"     • {sentiment}: {count:,} ({percentage:.1f}%)")
    
    # Check balance
    if abs(sentiment_counts.iloc[0] - sentiment_counts.iloc[1]) < len(data) * 0.05:
        print_success("Dataset is well-balanced")
    else:
        print("  ⚠️  Dataset may be imbalanced")
    
    # Text statistics
    print("\n Text Statistics:")
    data['review_length'] = data['review'].apply(lambda x: len(str(x).split()))
    print(f"     • Avg review length: {data['review_length'].mean():.0f} words")
    print(f"     • Min length: {data['review_length'].min()} words")
    print(f"     • Max length: {data['review_length'].max()} words")
    
    print_success("EDA complete")
    return data

df = perform_eda(df)

# ============================================================================
# STEP 5: GENERATE VISUALIZATIONS
# ============================================================================

def create_visualizations(data):
    """Generate word clouds and distribution charts"""
    print_section("Step 5: Generating Visualizations")
    
    try:
        print("  Creating visualizations (this may take 1-2 minutes)...")
        generate_visualizations(data, output_dir=VISUALIZATIONS_DIR)
        print_success(f"Visualizations saved to: {VISUALIZATIONS_DIR}")
        
        # List generated files
        files = os.listdir(VISUALIZATIONS_DIR)
        for file in files:
            print(f"     ✓ {file}")
    
    except Exception as e:
        print(f"  Warning: Could not generate visualizations: {e}")

create_visualizations(df)

# ============================================================================
# STEP 6: DATA PREPROCESSING
# ============================================================================

def preprocess_data(data):
    """Clean and prepare text for modeling"""
    print_section("Step 6: Preprocessing Text Data")
    
    print("  Cleaning reviews (this may take 3-5 minutes)...")
    print("  Progress: ", end="", flush=True)
    
    cleaned_reviews = []
    total = len(data)
    
    for idx, review in enumerate(data['review']):
        # Apply preprocessing
        cleaned = preprocess_text(review)
        cleaned_reviews.append(cleaned)
        
        # Progress indicator
        if (idx + 1) % (total // 10) == 0:
            print(f"{((idx + 1) // (total // 10)) * 10}%", end=" ", flush=True)
    
    print("100% ✓")
    
    # Create processed dataframe
    df_processed = pd.DataFrame({
        'cleaned_review': cleaned_reviews,
        'sentiment': data['sentiment'].values
    })
    
    # Save processed data
    df_processed.to_csv(PROCESSED_DATA_PATH, index=False)
    print_success(f"Processed data saved to: {PROCESSED_DATA_PATH}")
    
    # Show examples
    print("\n  Examples of preprocessing:")
    for i in range(2):
        print(f"\n     Original:  {data['review'].iloc[i][:80]}...")
        print(f"     Cleaned:   {cleaned_reviews[i][:80]}...")
    
    return df_processed

df_processed = preprocess_data(df)

# ============================================================================
# STEP 7: SPLIT DATA
# ============================================================================

def split_dataset(data):
    """Split into train and test sets"""
    print_section("Step 7: Train-Test Split")
    
    from sklearn.model_selection import train_test_split
    
    X = data['cleaned_review']
    y = data['sentiment']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=TEST_SIZE, 
        random_state=RANDOM_STATE,
        stratify=y  # Maintain sentiment distribution
    )
    
    print(f"  • Total samples: {len(data):,}")
    print(f"  • Training set: {len(X_train):,} ({(1-TEST_SIZE)*100:.0f}%)")
    print(f"  • Test set: {len(X_test):,} ({TEST_SIZE*100:.0f}%)")
    print(f"\n  Training set sentiment distribution:")
    print(f"     {y_train.value_counts().to_dict()}")
    
    print_success("Data split complete")
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = split_dataset(df_processed)

# ============================================================================
# STEP 8: FEATURE ENGINEERING
# ============================================================================

def create_features(X_train, X_test):
    """Convert text to TF-IDF features"""
    print_section("Step 8: Feature Engineering (TF-IDF Vectorization)")
    
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    print(f"  Vectorizing text with max {MAX_FEATURES:,} features...")
    
    # Create and fit vectorizer
    vectorizer = TfidfVectorizer(
        max_features=MAX_FEATURES,
        stop_words='english',
        ngram_range=(1, 2),  # Unigrams and bigrams
        min_df=2,            # Ignore terms appearing in < 2 documents
        max_df=0.95          # Ignore terms appearing in > 95% documents
    )
    
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    print_success("Vectorization complete")
    print(f"  • Vocabulary size: {len(vectorizer.get_feature_names_out()):,}")
    print(f"  • Training features shape: {X_train_vec.shape}")
    print(f"  • Test features shape: {X_test_vec.shape}")
    print(f"  • Sparsity: {1 - (X_train_vec.nnz / (X_train_vec.shape[0] * X_train_vec.shape[1])) * 100:.2f}%")
    
    # Save vectorizer
    with open(os.path.join(MODELS_DIR, 'tfidf_vectorizer.pkl'), 'wb') as f:
        pickle.dump(vectorizer, f)
    print(f"  • Vectorizer saved to: {MODELS_DIR}tfidf_vectorizer.pkl")
    
    return X_train_vec, X_test_vec, vectorizer

X_train_vec, X_test_vec, vectorizer = create_features(X_train, X_test)

# ============================================================================
# STEP 9: TRAIN MODELS
# ============================================================================

def train_sentiment_models(X_train, y_train):
    """Train three classification models"""
    print_section("Step 9: Training Machine Learning Models")
    
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.svm import LinearSVC
    
    models = {}
    
    # Model 1: Logistic Regression
    print("\n Training Logistic Regression...")
    lr_model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, n_jobs=-1)
    lr_model.fit(X_train, y_train)
    models['Logistic Regression'] = lr_model
    print("     ✓ Complete")
    
    # Model 2: Naïve Bayes
    print(" Training Multinomial Naïve Bayes...")
    nb_model = MultinomialNB()
    nb_model.fit(X_train, y_train)
    models['Naïve Bayes'] = nb_model
    print("     ✓ Complete")
    
    # Model 3: SVM
    print(" Training Linear SVM...")
    svm_model = LinearSVC(max_iter=2000, random_state=RANDOM_STATE)
    svm_model.fit(X_train, y_train)
    models['SVM'] = svm_model
    print("     ✓ Complete")
    
    print_success(f"All {len(models)} models trained successfully")
    return models

trained_models = train_sentiment_models(X_train_vec, y_train)

# ============================================================================
# STEP 10: EVALUATE MODELS
# ============================================================================

def evaluate_sentiment_models(models, X_test, y_test):
    """Evaluate models on test set"""
    print_section("Step 10: Evaluating Models")
    
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
    
    results = []
    
    print("\n Model Performance Metrics:\n")
    
    for model_name, model in models.items():
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, pos_label='positive')
        recall = recall_score(y_test, y_pred, pos_label='positive')
        f1 = f1_score(y_test, y_pred, pos_label='positive')
        
        results.append({
            'Model': model_name,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1
        })
        
        # Print results
        print(f"  {model_name}:")
        print(f"     • Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"     • Precision: {precision:.4f}")
        print(f"     • Recall:    {recall:.4f}")
        print(f"     • F1-Score:  {f1:.4f}\n")
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Find best model
    best_model_idx = results_df['Accuracy'].idxmax()
    best_model_name = results_df.loc[best_model_idx, 'Model']
    best_accuracy = results_df.loc[best_model_idx, 'Accuracy']
    
    print(f" Best Model: {best_model_name} ({best_accuracy*100:.2f}% accuracy)\n")
    
    # Save results
    results_df.to_csv(os.path.join(RESULTS_DIR, 'model_performance.csv'), index=False)
    print_success(f"Results saved to: {RESULTS_DIR}model_performance.csv")
    
    return results_df, best_model_name

results_df, best_model = evaluate_sentiment_models(trained_models, X_test_vec, y_test)

# ============================================================================
# STEP 11: SAVE MODELS
# ============================================================================

def save_trained_models(models):
    """Save models to pickle files"""
    print_section("Step 11: Saving Trained Models")
    
    for model_name, model in models.items():
        # Create filename
        filename = model_name.lower().replace(' ', '_').replace('ï', 'i') + '.pkl'
        filepath = os.path.join(MODELS_DIR, filename)
        
        # Save model
        with open(filepath, 'wb') as f:
            pickle.dump(model, f)
        
        print(f"  ✓ {model_name} → {filepath}")
    
    print_success("All models saved successfully")

save_trained_models(trained_models)

# ============================================================================
# STEP 12: MAKE PREDICTIONS ON NEW REVIEWS
# ============================================================================

def predict_sentiment(review_text, model, vectorizer):
    """Make sentiment prediction on new review"""
    # Preprocess
    cleaned = preprocess_text(review_text)
    
    # Vectorize
    X_new = vectorizer.transform([cleaned])
    
    # Predict
    prediction = model.predict(X_new)[0]
    confidence = model.predict_proba(X_new).max()
    
    return prediction, confidence

print_section("Step 12: Testing Predictions on Sample Reviews")

# Load best model
best_model_path = os.path.join(MODELS_DIR, best_model.lower().replace(' ', '_').replace('ï', 'i') + '.pkl')
best_model_obj = pickle.load(open(best_model_path, 'rb'))

sample_reviews = [
    "This movie was absolutely amazing! Best film I've ever seen.",
    "Terrible waste of time. Boring and poorly acted.",
    "It was okay, nothing special but watchable enough.",
    "Brilliant direction and outstanding performances throughout!",
    "One of the worst movies I've watched. Complete disappointment."
]

print(f"\n  Using: {best_model}\n")

for i, review in enumerate(sample_reviews, 1):
    prediction, confidence = predict_sentiment(review, best_model_obj, vectorizer)
    print(f"  {i}. \"{review}\"")
    print(f"     Prediction: {prediction.upper()} (confidence: {confidence*100:.2f}%)\n")

print_success("Predictions complete")

# ============================================================================
# STEP 13: FINAL SUMMARY & NEXT STEPS
# ============================================================================

print_section("PIPELINE COMPLETE! ✅")

summary = f"""
╔════════════════════════════════════════════════════════════════════════════╗
║                        PROJECT SUMMARY                                    ║
╚════════════════════════════════════════════════════════════════════════════╝

 DATA:
   • Raw dataset: {len(df):,} reviews
   • Training set: {len(X_train):,} reviews (80%)
   • Test set: {len(X_test):,} reviews (20%)

 MODELS TRAINED:
   • Logistic Regression  → {results_df[results_df['Model']=='Logistic Regression']['Accuracy'].values[0]*100:.2f}% accuracy
   • Naïve Bayes          → {results_df[results_df['Model']=='Naïve Bayes']['Accuracy'].values[0]*100:.2f}% accuracy
   • SVM                  → {results_df[results_df['Model']=='SVM']['Accuracy'].values[0]*100:.2f}% accuracy

 BEST MODEL: {best_model}
   ({results_df[results_df['Model']==best_model]['Accuracy'].values[0]*100:.2f}% accuracy)

 SAVED ARTIFACTS:
   Models:       {MODELS_DIR}
   Results:      {RESULTS_DIR}model_performance.csv
   Visualizations: {VISUALIZATIONS_DIR}

"""
summary_path = os.path.join(RESULTS_DIR, 'pipeline_summary.txt')

# Write summary to file with UTF-8 encoding
with open(summary_path, 'w', encoding='utf-8') as f:
    f.write(summary)

print(f"✓ Summary saved to: {summary_path}")
