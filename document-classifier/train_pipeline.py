"""
Document Classification Training Pipeline
==========================================
This script trains a document classifier using TF-IDF vectorization 
and Logistic Regression, then saves the complete pipeline for later use.
"""

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

# ============================================================================
# STEP 1: Define Dummy Dataset
# ============================================================================
# Simulate a corpus of documents across 3 departments

documents = [
    # HR Documents
    "employee performance review and annual appraisal discussion",
    "recruitment process for new candidates and hiring procedures",
    "leave policy and vacation request form submission",
    "employee benefits enrollment and insurance coverage details",
    "workplace harassment complaint and investigation procedures",
    "onboarding new employee orientation and training schedule",
    "salary increment and compensation review process",
    "employee resignation letter and exit interview",
    "performance improvement plan and corrective action",
    "diversity and inclusion training workshop",
    
    # Finance Documents
    "quarterly financial report and revenue analysis",
    "budget allocation for the next fiscal year",
    "expense reimbursement claim and invoice processing",
    "accounts payable and vendor payment schedule",
    "tax filing requirements and compliance documentation",
    "investment portfolio review and asset management",
    "profit and loss statement for the quarter",
    "accounts receivable and customer payment tracking",
    "financial audit preparation and documentation",
    "capital expenditure approval and authorization",
    
    # IT Documents
    "network security incident and vulnerability assessment",
    "software development lifecycle and project management",
    "database backup and disaster recovery procedures",
    "user access permissions and authentication system",
    "server maintenance and system upgrade schedule",
    "cybersecurity policy and data protection measures",
    "helpdesk ticket resolution and technical support",
    "cloud infrastructure deployment and configuration",
    "application programming interface documentation",
    "software bug tracking and issue resolution"
]

# Corresponding labels for each document
labels = [
    "HR", "HR", "HR", "HR", "HR", "HR", "HR", "HR", "HR", "HR",
    "Finance", "Finance", "Finance", "Finance", "Finance", 
    "Finance", "Finance", "Finance", "Finance", "Finance",
    "IT", "IT", "IT", "IT", "IT", "IT", "IT", "IT", "IT", "IT"
]

print(f"Total documents: {len(documents)}")
print(f"Classes: {set(labels)}")
print(f"Class distribution: HR={labels.count('HR')}, "
      f"Finance={labels.count('Finance')}, IT={labels.count('IT')}\n")

# ============================================================================
# STEP 2: Split Data into Training and Testing Sets
# ============================================================================
# Use 80-20 split with stratification to maintain class balance

X_train, X_test, y_train, y_test = train_test_split(
    documents, 
    labels, 
    test_size=0.2, 
    random_state=42,
    stratify=labels  # Ensures balanced representation in train/test
)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}\n")

# ============================================================================
# STEP 3: Create ML Pipeline
# ============================================================================
# Pipeline combines preprocessing (TF-IDF) and model (Logistic Regression)

pipeline = Pipeline([
    # Stage 1: Convert text documents to TF-IDF feature vectors
    ('tfidf', TfidfVectorizer(
        max_features=100,        # Limit vocabulary to top 100 terms
        ngram_range=(1, 2),      # Use unigrams and bigrams
        stop_words='english',    # Remove common English stop words
        lowercase=True           # Normalize text to lowercase
    )),
    
    # Stage 2: Train Logistic Regression classifier
    ('classifier', LogisticRegression(
        max_iter=1000,           # Maximum iterations for convergence
        random_state=42,         # For reproducibility
        solver='lbfgs',          # Optimization algorithm
        multi_class='multinomial' # Handle 3-class classification
    ))
])

print("Pipeline created with TF-IDF Vectorizer and Logistic Regression")

# ============================================================================
# STEP 4: Train the Model
# ============================================================================
print("\nTraining the model...")
pipeline.fit(X_train, y_train)
print("Training complete!\n")

# ============================================================================
# STEP 5: Evaluate Model Performance
# ============================================================================
# Make predictions on the test set
y_pred = pipeline.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2%}\n")

# Display detailed classification report
print("Classification Report:")
print("=" * 60)
print(classification_report(y_test, y_pred))

# ============================================================================
# STEP 6: Save the Complete Pipeline
# ============================================================================
# Save both the vectorizer and classifier in a single pipeline object

model_filename = 'model_pipeline.joblib'
joblib.dump(pipeline, model_filename)
print(f"\nPipeline saved successfully to '{model_filename}'")
print("\nYou can now use 'predict_service.py' to classify new documents!")
