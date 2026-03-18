# FinTech Sentinel: High-Precision Fraud & Risk Engine

Engineered a multi-layered fraud detection system for a credit card dataset, transitioning from a 0% baseline detection to a high-impact model that secured 83.25% of total fraudulent capital ($3.32M) while maintaining operational feasibility.

[![kaggle](https://img.shields.io/badge/Kaggle-Profile-blue?logo=kaggle&logoColor=white)(https://www.kaggle.com/code/auro15/fraud-analytics)]

------------------------------
## Project Pipeline

```
Raw Dataset (Kaggle)
        │
        ▼
Exploratory Data Analysis
        │
        ▼
Feature Engineering
        │
        ▼
Fraud Risk Alert System (65%)
```
--------------------------------
## Key Performance Indicators (KPIs)

* Economic Protection Rate: 83.25% ($3,320,086.29 saved).
* Fraud Detection Rate (Volume): 66.20% (4,969 cases caught).
* Precision Ratio: 1:5.4 (18.55% Precision—significantly higher than baseline random flagging).
* Operational Optimization: Reduced False Positives by 92% (from 247k to 21k) through "Double-Lock" merchant verification and velocity filtering.

------------------------------
## Technical Implementation & Logic

* Feature Engineering: Identified that "Distance" was a dead feature (~47-mile fraud radius), pivoting the strategy toward Temporal (Night-Owl) and Categorical (Digital Whale) patterns.
* Velocity Burst Logic: Developed a "burst" filter identifying transactions within <10-minute windows, capturing "strip-mining" behavior at local merchants.
* The "Sniper" Rule: Built a high-risk merchant blacklist (5+ fraud incidents) and combined it with value thresholds to isolate "playgrounds" for stolen credentials.
* Victim Profiling: Analyzed demographic "hotspots" (High-risk ages and specific professions like Materials Engineers) to add a surgical layer of protection for vulnerable segments.

------------------------------
## Strategic Justification for ML Transition

* The Rule-Based Ceiling: Identified the statistical plateau at 66% detection where further static rules caused exponential false positive growth.
* Path to 95%: Proposed a transition to Random Forest/XGBoost to resolve "chameleon" fraud (low-value, daytime transactions) through multi-dimensional correlation that static "If/Then" logic cannot capture.


