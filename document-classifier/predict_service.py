"""
Document Classification Prediction Service
==========================================
This script loads a pre-trained document classification pipeline
and uses it to classify new, unseen documents.
"""

import joblib
import numpy as np

# ============================================================================
# STEP 1: Load the Trained Pipeline
# ============================================================================
print("Loading the trained model pipeline...")

try:
    pipeline = joblib.load('model_pipeline.joblib')
    print("✓ Pipeline loaded successfully!\n")
except FileNotFoundError:
    print("ERROR: 'model_pipeline.joblib' not found.")
    print("Please run 'train_pipeline.py' first to train and save the model.")
    exit(1)

# ============================================================================
# STEP 2: Define New, Unseen Documents for Classification
# ============================================================================
# These are test documents that the model has never seen before

new_documents = [
    "submit expense report for business travel reimbursement",
    "employee attendance tracking and time management system",
    "firewall configuration and network intrusion detection",
    "annual budget forecast and cost analysis review",
    "password reset request and account security verification",
    "employee termination process and final settlement"
]

print("New documents to classify:")
print("=" * 70)
for idx, doc in enumerate(new_documents, 1):
    print(f"{idx}. {doc}")
print("\n")

# ============================================================================
# STEP 3: Make Predictions
# ============================================================================
# The pipeline automatically handles vectorization and classification

predictions = pipeline.predict(new_documents)

# Get prediction probabilities for confidence scores
prediction_probabilities = pipeline.predict_proba(new_documents)

print("Classification Results:")
print("=" * 70)

for idx, (doc, pred, probs) in enumerate(zip(new_documents, predictions, prediction_probabilities), 1):
    # Get the confidence score for the predicted class
    confidence = np.max(probs) * 100
    
    print(f"\nDocument {idx}:")
    print(f"Text: {doc}")
    print(f"Predicted Class: {pred}")
    print(f"Confidence: {confidence:.2f}%")
    
    # Show probability distribution across all classes
    print(f"Probability Distribution:")
    for class_name, prob in zip(pipeline.classes_, probs):
        print(f"  {class_name}: {prob*100:.2f}%")

print("\n" + "=" * 70)
print("Classification complete!")

# ============================================================================
# STEP 4: Single Document Prediction Function (Utility)
# ============================================================================
def classify_document(text):
    """
    Utility function to classify a single document.
    
    Args:
        text (str): The document text to classify
    
    Returns:
        dict: Contains predicted class and confidence score
    """
    prediction = pipeline.predict([text])[0]
    probabilities = pipeline.predict_proba([text])[0]
    confidence = np.max(probabilities) * 100
    
    return {
        'predicted_class': prediction,
        'confidence': confidence,
        'all_probabilities': dict(zip(pipeline.classes_, probabilities))
    }

# Example usage of the utility function
print("\n" + "=" * 70)
print("Testing single document classification function:")
print("=" * 70)

test_doc = "cloud server deployment and infrastructure management"
result = classify_document(test_doc)

print(f"\nDocument: {test_doc}")
print(f"Predicted Class: {result['predicted_class']}")
print(f"Confidence: {result['confidence']:.2f}%")
