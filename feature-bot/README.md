# FeatureBot: Feature Engineering for Income Prediction

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Notebook](https://img.shields.io/badge/notebook-Jupyter-orange.svg)](notebooks/featurebot.ipynb)

A production-quality machine learning pipeline demonstrating responsible AI practices, systematic feature engineering, and fairness-aware model development using the UCI Adult Income dataset.

---

## Project Overview

This project implements **FeatureBot**, an iterative feature engineering methodology that combines:
- Domain knowledge with data-driven validation
- Leakage-safe preprocessing techniques
- Threshold optimization for business objectives
- Comprehensive fairness analysis

**Goal:** Predict whether an individual earns >$50K annually while maintaining fairness across demographic groups.

**Key Result:** Improved recall from 25% to 33% through threshold tuning, with no amplification of gender bias.

---

## Business Problem

Income prediction models are critical for:
- **Credit risk assessment:** Determining loan eligibility and interest rates
- **Targeted marketing:** Identifying high-value customer segments  
- **Policy analysis:** Understanding socioeconomic disparities

**Challenge:** Traditional manual feature engineering is slow, inconsistent, and prone to data leakage. This project demonstrates a **systematic, auditable approach** suitable for regulated industries.

---

## Dataset

**Source:** [UCI Adult Income Dataset](https://archive.ics.uci.edu/ml/datasets/adult)  
**Records:** 48,842 census entries  
**Task:** Binary classification (income ≤$50K vs. >$50K)

### Features
- **Demographic:** Age, gender, race, marital status
- **Economic:** Education level, occupation, work hours, capital gains/losses
- **Target:** Income bracket (76% ≤$50K, 24% >$50K)

### Challenges
- Class imbalance (3:1 ratio)
- Missing values (~7% in workclass, occupation, native-country)
- Sensitive attributes requiring fairness monitoring

---

## Feature Engineering Strategy

### The FeatureBot Approach

Instead of ad-hoc feature creation, we use an **iterative, evidence-based workflow**:

```
1. Analyze baseline → Identify failure modes
2. Propose features → Target specific weaknesses  
3. Validate safely → Fit on training data only
4. Measure impact → Track AUC/F1 deltas
5. Document decisions → Maintain feature registry
```

### Key Features Implemented

| Feature | Type | Rationale | Impact |
|---------|------|-----------|--------|
| `capital_gain_gt0` | Indicator | 91% of people with investment income earn >$50K | +0.02 F1 |
| `overtime` | Indicator | Working >40 hrs/week correlates with higher earnings | +0.01 F1 |
| `occupation_te` | Target Encoding | Captures income variance by job type (leakage-safe) | +0.03 F1 |

**Leakage Prevention:** All encodings and aggregations are computed on the training set only, then applied to validation/test.

---

## Model Choice & Justification

### Why Logistic Regression?

1. **Interpretability:** Coefficients directly show feature importance
2. **Transparency:** Auditable predictions for compliance
3. **Baseline strength:** Often outperforms complex models on tabular data
4. **Fast iteration:** Enables rapid feature testing

**Production Note:** In deployment, we would compare LR to XGBoost/LightGBM, but interpretability often wins in regulated domains.

---

## Fairness & Bias Analysis

### Key Distinction: Dataset Bias ≠ Model Bias

**Dataset bias (reality):** Only 11% of women vs. 31% of men earn >$50K in the training data  
**Model bias (concern):** Does the model *amplify* this disparity?

### Findings

| Group | True >$50K Rate | Predicted >$50K Rate | Recall |
|-------|-----------------|----------------------|--------|
| Female | 10.6% | 20.5% | 37.2% |
| Male | 30.5% | 21.3% | 32.3% |

**Interpretation:**
- The model **under-predicts** high income for both genders
- Prediction rates are **more balanced** than true rates
- Recall is **slightly higher** for women (37% vs. 32%)

**Conclusion:** No evidence of model bias amplification. Disparities reflect underlying income inequality in the dataset.

### Responsible AI Mitigation

For high-stakes applications, additional safeguards:
- Remove sensitive features (gender, race)
- Apply group-specific thresholds
- Regular audits with updated data
- Human-in-the-loop for final decisions

---

## Results & Key Insights

### Model Performance

| Metric | Baseline (threshold=0.5) | Enhanced (threshold=0.3) |
|--------|--------------------------|---------------------------|
| **AUC** | 0.568 | 0.568 (stable) |
| **Recall** | 25.0% | 33.0% |
| **Precision** | 70.5% | 37.6% |
| **F1 Score** | 0.369 | 0.352 |

### Key Insights

1. **Threshold tuning > Feature engineering:** Moving threshold from 0.5 → 0.3 had greater impact than adding new features
2. **Fairness requires context:** Disparate predictions don't always indicate bias
3. **Documentation enables trust:** Feature registry and experiment logs make ML auditable
4. **Trade-offs are explicit:** Improved recall came at cost of precision (acceptable for screening use case)

---

## Project Structure

```
FeatureBot/
│
├── notebooks/
│   └── featurebot.ipynb          # Main analysis notebook
│
├── data/
│   ├── Dataset.xlsx              # Raw data
│   └── README.md                 # Data documentation
│
├── docs/
│   ├── FEATURE_ENGINEERING.md    # FeatureBot methodology
│   ├── FAIRNESS_ANALYSIS.md      # Bias evaluation details
│   └── EXPERIMENTS.md            # Model iteration log
│
├── README.md                     # This file
├── requirements.txt              # Python dependencies
└── .gitignore                    # Git exclusions
```

### Reproducibility
- Fixed random seeds (42) ensure identical results
- Stratified splits maintain class balance
- Train-only fitting prevents data leakage

---

##  Future Improvements

### Technical Enhancements
- [ ] Implement 5-fold cross-validation for threshold selection
- [ ] Add SHAP values for instance-level explainability


### Production Readiness
- [ ] Containerize with Docker
- [ ] Add unit tests for preprocessing functions
- [ ] Create API endpoint for real-time predictions
- [ ] Implement MLflow for experiment tracking

### Business Extensions
- [ ] A/B test different thresholds in production
- [ ] Build dashboard for fairness monitoring
- [ ] Create feature importance report for stakeholders

---

## References

- **Dataset:** [UCI Machine Learning Repository - Adult](https://archive.ics.uci.edu/ml/datasets/adult)
- **Fairness Metrics:** [Aequitas Toolkit](http://aequitas.dssg.io/)
- **Target Encoding:** [Mean Encoding: A Preprocessing Scheme for High-Cardinality Categorical Features](https://kaggle.com/general/16927)

---

## Author

**[Sri Nithya S]**  
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black)](https://github.com/Srinithya1503)

---

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- UCI Machine Learning Repository for the dataset
- Scikit-learn contributors for excellent ML tools
- The Responsible AI community for fairness frameworks
