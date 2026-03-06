# Feature Engineering Methodology

## The FeatureBot Philosophy

FeatureBot is a **systematic, evidence-based approach** to feature engineering that emphasizes:
- Reproducibility over trial-and-error
- Measurement over intuition
- Safety over speed

---

## Core Principles

### 1. Leakage Prevention

**The Problem:** If you compute statistics (means, frequencies, encodings) using the entire dataset, your validation metrics will be artificially inflated.

**The Solution:** Fit all transformations on training data only.

```python
# ❌ WRONG: Uses validation data in encoding
all_data = pd.concat([X_train, X_val])
occ_mean = all_data.groupby('occupation')['income'].mean()

# ✅ CORRECT: Train-only encoding
train_with_target = X_train.copy()
train_with_target['income'] = y_train
occ_mean = train_with_target.groupby('occupation')['income'].mean()

# Apply to both train and validation
X_train['occupation_te'] = X_train['occupation'].map(occ_mean)
X_val['occupation_te'] = X_val['occupation'].map(occ_mean)
```

---

### 2. Feature Validation Framework

Every feature must pass three tests:

| Test | Criterion | Action if Failed |
|------|-----------|------------------|
| **Leakage Check** | No validation data used in computation | Reject immediately |
| **Impact Measure** | ≥0.01 improvement in AUC or F1 | Deprioritize |
| **Stability Test** | Consistent across CV folds | Flag for review |

---

### 3. Feature Registry

Every engineered feature is documented with:

```python
{
    "name": "capital_gain_gt0",
    "definition": "Binary indicator: 1 if capital-gain > 0, else 0",
    "type": "indicator",
    "dependencies": ["capital-gain"],
    "leakage_risk": "low",
    "added_in_cycle": 1,
    "delta_auc": +0.012,
    "delta_f1": +0.018,
    "notes": "Addresses false negatives for high-earners with investment income"
}
```

---

## Feature Types & When to Use Them

### Indicator Features
**Purpose:** Capture non-linear relationships  
**Example:** `overtime = (hours_per_week > 40)`  
**When:** Binary splits in the data (e.g., 91% of people working >40 hrs earn more)

### Target Encoding
**Purpose:** Compress high-cardinality categoricals  
**Example:** `occupation_te = mean(income | occupation)`  
**When:** 50+ unique values in a categorical feature

### Interaction Features
**Purpose:** Capture synergies between features  
**Example:** `age_x_education = age * education_num`  
**When:** Domain knowledge suggests multiplicative effects

### Binning
**Purpose:** Handle outliers and non-linearities  
**Example:** `age_bin = cut(age, bins=[0, 25, 40, 60, 100])`  
**When:** Histograms show distinct modes or thresholds

---

## The FeatureBot Workflow

```
┌─────────────────────┐
│  1. Analyze Errors  │ → Which records are misclassified?
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  2. Propose Feature │ → What pattern distinguishes them?
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  3. Implement Safely│ → Train-only fitting
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  4. Measure Impact  │ → AUC/F1 delta > threshold?
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  5. Update Registry │ → Document decision
└─────────────────────┘
```

---

## Common Pitfalls & How to Avoid Them

| Pitfall | Example | Fix |
|---------|---------|-----|
| **Target leakage** | Using `income` in feature creation | Always drop target before feature engineering |
| **Validation leakage** | Fitting imputers on all data | Use `.fit(X_train)` then `.transform(X_val)` |
| **Overfitting** | Adding 100 features without validation | Use feature selection (univariate, importance) |
| **Correlation** | Including both `education` and `education_num` | Check VIF, remove redundant features |

---

## Success Metrics

A feature engineering cycle is successful if:
1. ✅ No data leakage detected (manual code review)
2. ✅ Validation AUC/F1 improves by ≥0.01
3. ✅ Test performance remains stable (no overfitting)
4. ✅ Feature registry is updated with rationale

---

## Lessons from This Project

1. **Threshold tuning beats feature engineering:** Moving from 0.5 → 0.3 improved recall by 8pp, more than any single feature
2. **Target encoding requires discipline:** Easy to leak, but powerful when done correctly
3. **Simpler features win:** `capital_gain_gt0` outperformed complex polynomial interactions
4. **Documentation enables trust:** Feature registry made audit and iteration straightforward

---

## Further Reading

- [Kaggle Feature Engineering Guide](https://www.kaggle.com/learn/feature-engineering)
- [Mean Encoding Best Practices](https://machinelearningmastery.com/target-encoding-for-categorical-variables/)
- [Preventing Data Leakage](https://towardsdatascience.com/data-leakage-in-machine-learning-10bdd3eec742)