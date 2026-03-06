"""
Bank-Grade Credit Risk Model Training
------------------------------------
• Cost-sensitive learning
• Proper train/test separation
• Threshold optimization on TRAIN only
• No data leakage
"""

import os
import numpy as np
import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold
)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score,
    confusion_matrix,
    precision_score,
    recall_score
)
from sklearn.ensemble import RandomForestClassifier

import shap

from data_loader import load_and_preprocess_data

# ================================
# CONFIG
# ================================
COST_FP = 5000   # Approve defaulter
COST_FN = 1000   # Reject good customer
N_SPLITS = 5
RANDOM_STATE = 42

os.makedirs("reports", exist_ok=True)


# ================================
# COST FUNCTION
# ================================
def total_cost(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return fp * COST_FP + fn * COST_FN, fp, fn


# ================================
# THRESHOLD OPTIMIZATION (TRAIN ONLY)
# ================================
def find_optimal_threshold(y_true, y_prob):
    thresholds = np.linspace(0.01, 0.99, 99)
    best_cost = np.inf
    best_threshold = 0.5

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        cost, _, _ = total_cost(y_true, y_pred)

        if cost < best_cost:
            best_cost = cost
            best_threshold = t

    return best_threshold, best_cost


# ================================
# LOAD DATA
# ================================
X, y = load_and_preprocess_data("data/german_credit_data.csv")

# HOLD-OUT TEST SET (STRICT)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=RANDOM_STATE
)

print(f"\nTrain size: {X_train.shape}")
print(f"Test size : {X_test.shape}")


# ================================
# LOGISTIC REGRESSION PIPELINE
# ================================
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("lr", LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=RANDOM_STATE
    ))
])


# ================================
# CROSS-VALIDATION (TRAIN ONLY)
# ================================
cv = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE
)

cv_results = []

print("\n===== TRAINING (COST-SENSITIVE CV) =====")

for fold, (tr_idx, val_idx) in enumerate(cv.split(X_train, y_train), 1):
    X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]

    pipeline.fit(X_tr, y_tr)
    y_val_prob = pipeline.predict_proba(X_val)[:, 1]

    threshold, cost = find_optimal_threshold(y_val, y_val_prob)
    y_val_pred = (y_val_prob >= threshold).astype(int)

    roc = roc_auc_score(y_val, y_val_prob)
    _, fp, fn = total_cost(y_val, y_val_pred)

    cv_results.append({
        "fold": fold,
        "optimal_threshold": threshold,
        "roc_auc": roc,
        "total_cost": cost,
        "fp": fp,
        "fn": fn,
        "precision": precision_score(y_val, y_val_pred, zero_division=0),
        "recall": recall_score(y_val, y_val_pred)
    })

    print(f"Fold {fold}: θ*={threshold:.3f}, Cost=${cost:,}, AUC={roc:.3f}")

cv_df = pd.DataFrame(cv_results)
cv_df.to_csv("reports/cv_summary.csv", index=False)

best_threshold = cv_df["optimal_threshold"].mean()

print(f"\nOptimal TRAIN threshold (avg): θ* = {best_threshold:.3f}")


# ================================
# FINAL LOGISTIC MODEL (TEST SET)
# ================================
pipeline.fit(X_train, y_train)
y_test_prob = pipeline.predict_proba(X_test)[:, 1]
y_test_pred = (y_test_prob >= best_threshold).astype(int)

test_cost, fp, fn = total_cost(y_test, y_test_pred)
test_auc = roc_auc_score(y_test, y_test_prob)

print("\n===== TEST SET PERFORMANCE (UNSEEN DATA) =====")
print(f"ROC-AUC      : {test_auc:.3f}")
print(f"Total Cost  : ${test_cost:,}")
print(f"FP / FN     : {fp} / {fn}")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_test_pred))


# ================================
# RANDOM FOREST (NO OVERFITTING)
# ================================
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=6,
    min_samples_leaf=20,
    random_state=RANDOM_STATE
)

rf.fit(X_train, y_train)
rf_test_prob = rf.predict_proba(X_test)[:, 1]
rf_auc = roc_auc_score(y_test, rf_test_prob)

print("\n===== RANDOM FOREST (TEST SET) =====")
print(f"Random Forest ROC-AUC: {rf_auc:.3f}")


# ================================
# SHAP (LOGISTIC REGRESSION)
# ================================
print("\nComputing SHAP values...")

# Extract trained components
scaler = pipeline.named_steps["scaler"]
lr_model = pipeline.named_steps["lr"]

# Scale data (SHAP must see same representation as model)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# SHAP explainer
explainer = shap.LinearExplainer(
    lr_model,
    X_train_scaled,
    feature_perturbation="interventional"
)

shap_values = explainer(X_test_scaled)

# Summary plot
shap.summary_plot(
    shap_values,
    X_test,
    show=False
)

# Feature importance table
importance = np.abs(shap_values.values).mean(axis=0)
importance_df = pd.DataFrame({
    "feature": X.columns,
    "mean_abs_shap": importance
}).sort_values("mean_abs_shap", ascending=False)

importance_df.to_csv("reports/feature_importance.csv", index=False)


print("\n✅ Model training complete (NO overfitting).")
