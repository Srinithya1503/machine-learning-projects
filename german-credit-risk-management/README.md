# German Credit Risk Management – Cost-Sensitive ML

## Project Overview
This project builds a **bank-grade credit risk classification system** using the German Credit dataset.
Instead of optimizing only for accuracy, the model minimizes **financial loss**, reflecting real-world
lending decisions.

The pipeline is designed with:
- Strict train/test separation
- Cost-sensitive threshold optimization
- Interpretable machine learning (SHAP)
- No data leakage or overfitting

---

## Business Objective
Banks face asymmetric risk:
- Approving a defaulter causes **large financial loss**
- Rejecting a good customer causes **smaller opportunity loss**

This project explicitly models that trade-off using a **cost matrix** and optimized decision thresholds.

---

## Dataset
- Source: UCI German Credit Dataset
- Samples: 1,000
- Features: 20 (financial + demographic)
- Target: Credit Default (Bad = 1, Good = 0)

---

## Modeling Approach
### Models Used
- Logistic Regression (cost-sensitive, interpretable)
- Random Forest (regularized benchmark)

### Key Techniques
- Stratified cross-validation
- Threshold optimization on training data only
- Cost-based evaluation
- SHAP-based feature importance

---

## Results (Unseen Test Set)
| Metric | Value |
|------|------|
| ROC-AUC | ~0.79 |
| False Positives | 2 |
| False Negatives | 54 |
| Optimal Threshold | ~0.92 |

The model prioritizes **risk reduction**, consistent with conservative banking policies.

---

## Explainability
SHAP analysis shows that:
- Checking account status
- Savings behavior
- Credit history
are the strongest predictors of default risk.

---

## Conclusion
This project demonstrates how machine learning can be aligned with **real financial decision-making**
while remaining interpretable, robust, and production-aware.

---

## Tech Stack
Python, Pandas, NumPy, Scikit-learn, SHAP

---

## Repository Structure
german-credit-risk-management/
├── data/
├── reports/
├── data_loader.py
├── train_model.py
└── README.md

## Author
[Sri Nithya S]
📧 Email: venkatsri1503@gmail.com


## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

UCI Machine Learning Repository for dataset provision
Scikit-learn community for robust ML infrastructure
SHAP library developers for interpretability framework

## Contact & Feedback
Questions about the project? Want to discuss credit risk modeling?
Open an issue or reach out directly - I'm always happy to chat about quantitative finance and machine learning!

Last Updated: January 2026
Version: 1.0.0
