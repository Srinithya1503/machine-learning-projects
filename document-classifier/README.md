# Document Classification Pipeline

A minimal, production-ready machine learning pipeline for classifying text documents into predefined categories using TF-IDF vectorization and Logistic Regression.

## Overview

This project demonstrates a complete end-to-end machine learning workflow for document classification:

- **Preprocessing**: Text vectorization using TF-IDF (Term Frequency-Inverse Document Frequency)
- **Model**: Logistic Regression classifier for multi-class classification
- **Pipeline**: Scikit-learn Pipeline combining preprocessing and modeling
- **Persistence**: Model serialization using joblib for deployment

The system classifies documents into three categories: **HR**, **Finance**, and **IT**.

## Features

- Complete ML lifecycle implementation (train, evaluate, save, load, predict)
- Clean separation between training and inference code
- Scikit-learn Pipeline for reproducible preprocessing
- Confidence scores and probability distributions for predictions
- Self-contained with dummy data (no external datasets required)
- Ready for production deployment

## Project Structure

```
document-classifier/
├── README.md                    # Project documentation
├── requirements.txt             # Python dependencies
├── train_pipeline.py            # Training script
├── predict_service.py           # Prediction/inference script
└── model_pipeline.joblib        # Saved model (generated after training)
```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone or download this repository**

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training the Model

Run the training script to build and save the classification model:

```bash
python train_pipeline.py
```

**What this does:**
- Loads 30 dummy documents across 3 categories (HR, Finance, IT)
- Splits data into training (80%) and testing (20%) sets
- Creates a TF-IDF + Logistic Regression pipeline
- Trains the model on the training set
- Evaluates performance on the test set
- Saves the complete pipeline to `model_pipeline.joblib`


### Making Predictions

Run the prediction service to classify new documents:

```bash
python predict_service.py
```

**What this does:**
- Loads the trained pipeline from `model_pipeline.joblib`
- Classifies 6 new unseen documents
- Displays predictions with confidence scores
- Shows probability distributions across all classes


### Custom Document Classification

You can modify `predict_service.py` to classify your own documents:

```python
# Add your custom document
custom_doc = "Your document text here"
result = classify_document(custom_doc)

print(f"Predicted Class: {result['predicted_class']}")
print(f"Confidence: {result['confidence']:.2f}%")
```

## Technical Details

### Model Architecture

**Pipeline Components:**

1. **TF-IDF Vectorizer**
   - Converts text documents into numerical feature vectors
   - Parameters:
     - `max_features=100`: Limits vocabulary to top 100 most important terms
     - `ngram_range=(1, 2)`: Uses both single words and word pairs
     - `stop_words='english'`: Removes common English words
     - `lowercase=True`: Normalizes text to lowercase

2. **Logistic Regression Classifier**
   - Multi-class classification algorithm
   - Parameters:
     - `max_iter=1000`: Maximum training iterations
     - `solver='lbfgs'`: Optimization algorithm
     - `multi_class='multinomial'`: Handles 3-class classification

### Dataset

The dummy dataset contains 30 documents:
- **HR**: 10 documents (employee management, recruitment, policies)
- **Finance**: 10 documents (budgets, expenses, financial reports)
- **IT**: 10 documents (security, infrastructure, technical support)

### Performance Metrics

The model is evaluated using:
- **Accuracy**: Overall percentage of correct predictions
- **Precision**: Ratio of true positives to predicted positives
- **Recall**: Ratio of true positives to actual positives
- **F1-Score**: Harmonic mean of precision and recall

## Extending the Pipeline

### Using Real Data

Replace the dummy data in `train_pipeline.py`:

```python
# Load your data
import pandas as pd
df = pd.read_csv('your_documents.csv')
documents = df['text'].tolist()
labels = df['category'].tolist()
```

### Adding More Classes

Simply include additional document categories in your training data. The Logistic Regression model automatically handles any number of classes.

### Hyperparameter Tuning

Use GridSearchCV for optimization:

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'tfidf__max_features': [50, 100, 200],
    'classifier__C': [0.1, 1.0, 10.0]
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5)
grid_search.fit(X_train, y_train)
```

### Alternative Models

Replace Logistic Regression with other classifiers:

```python
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier

# Naive Bayes (fast, good for text)
('classifier', MultinomialNB())

# Random Forest (handles non-linear patterns)
('classifier', RandomForestClassifier(n_estimators=100))
```

## Dependencies

- **scikit-learn**: Machine learning library (pipeline, vectorizer, classifier)
- **pandas**: Data manipulation (optional, for loading CSV data)
- **joblib**: Model serialization and deserialization
- **numpy**: Numerical computations

## Contributing

To extend or improve this pipeline:

1. Add more sophisticated preprocessing (lemmatization, custom tokenization)
2. Implement cross-validation for robust evaluation
3. Add feature importance analysis
4. Create a REST API for model serving
5. Add logging and monitoring capabilities

## License

This project is provided as-is for educational and demonstration purposes.

---
