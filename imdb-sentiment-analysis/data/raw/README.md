# 📊 Data Directory

This folder contains all dataset files for the IMDb Sentiment Analysis project.

## 📁 Directory Structure

```
data/
├── raw/
│   └── imdb_dataset.csv          # Original dataset from Kaggle
├── processed/
│   └── cleaned_reviews.csv       # Preprocessed data (auto-generated)
└── README.md                      # This file
```

## 📥 Raw Data

### `imdb_dataset.csv`

**Location:** `data/raw/imdb_dataset.csv`

**Status:** ⚠️ NOT TRACKED in Git (large file)

**How to obtain:**

1. Visit [Kaggle - IMDb Movie Reviews Dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
2. Download `imdb_dataset.csv`
3. Place file in `data/raw/` folder

**File Details:**
- **Size:** ~130 MB
- **Format:** CSV
- **Rows:** 50,000 movie reviews
- **Columns:** 2
  - `review` (str): Text of the movie review (0-10,000+ characters)
  - `sentiment` (str): Label - either "positive" or "negative"

**Data Balance:** 
- Positive reviews: 25,000 (50%)
- Negative reviews: 25,000 (50%)
- ✓ Perfectly balanced dataset

## 🔧 Processed Data

### `cleaned_reviews.csv`

**Location:** `data/processed/cleaned_reviews.csv`

**Status:** ✅ AUTO-GENERATED (created by `sentiment_analysis.py`)

**How it's created:**

Run the main pipeline script:

```bash
python sentiment_analysis.py
```

**Preprocessing Steps Applied:**
1. Remove HTML tags (`<br/>`, etc.)
2. Convert to lowercase
3. Remove URLs
4. Remove special characters and punctuation
5. Tokenization
6. Remove stopwords
7. Lemmatization

## 📋 Data Specifications

### Text Statistics

| Metric | Value |
|--------|-------|
| Total Reviews | 50,000 |
| Avg Words per Review | 238 |
| Min Words | 10 |
| Max Words | 2,500+ |
| Average Characters | 1,300 |

## 🚀 Quick Start

### 1. Download Dataset

```bash
# Visit: https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
# Download imdb_dataset.csv and place in data/raw/
```

### 2. Run Pipeline

```bash
python sentiment_analysis.py
```

### 3. Check Processed Data

```python
import pandas as pd

df = pd.read_csv('data/processed/cleaned_reviews.csv')
print(df.head())
print(df.info())
```

## 📝 License

The IMDb dataset is provided by Kaggle and is intended for educational purposes only. Please refer to the dataset's license on Kaggle.
