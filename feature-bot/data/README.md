# Data Directory

## Dataset Information

**Name:** UCI Adult Income Dataset  
**Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/adult)  
**Alternative Source:** [Kaggle](https://www.kaggle.com/datasets/uciml/adult-census-income)

## File Format

The dataset should be placed in this directory as `Dataset.xlsx` (Excel format).

### Expected Columns

| Column Name | Type | Description |
|-------------|------|-------------|
| age | Integer | Age in years |
| workclass | Categorical | Employment type (Private, Self-emp, Gov, etc.) |
| fnlwgt | Integer | Census sampling weight |
| education | Categorical | Highest education level (HS-grad, Bachelors, etc.) |
| educational-num | Integer | Numeric encoding of education (1-16) |
| marital-status | Categorical | Marital status (Married, Divorced, etc.) |
| occupation | Categorical | Job type (Tech-support, Craft-repair, etc.) |
| relationship | Categorical | Family relationship (Husband, Wife, Own-child, etc.) |
| race | Categorical | Race (White, Black, Asian-Pac-Islander, etc.) |
| gender | Categorical | Male or Female |
| capital-gain | Integer | Investment income |
| capital-loss | Integer | Investment losses |
| hours-per-week | Integer | Weekly work hours |
| native-country | Categorical | Country of origin |
| income | Binary | Target variable (<=50K or >50K) |

## Data Characteristics

- **Records:** 48,842
- **Features:** 14 (6 numeric, 8 categorical)
- **Target Distribution:** 76% ≤$50K, 24% >$50K
- **Missing Values:** ~7% in workclass, occupation, native-country (encoded as "?")

## Download Instructions

1. **Option 1: UCI Repository**
   ```bash
   wget https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data
   ```

2. **Option 2: Kaggle** (requires Kaggle account)
   - Visit: https://www.kaggle.com/datasets/uciml/adult-census-income
   - Download `adult.csv`
   - Convert to Excel format if needed

3. **Option 3: Manual Download**
   - Download from the UCI link above
   - Save as `Dataset.xlsx` in this directory

## Data Preprocessing Notes

The notebook handles the following preprocessing steps:
- Replacing "?" with NaN
- Normalizing target labels (>50K, <=50K, >50K., <=50K.)
- Stratified train/validation/test split (60/20/20)
- Median imputation for numeric features
- Most-frequent imputation for categorical features

## Privacy & Ethics

- This is a publicly available dataset from the 1994 US Census
- Contains no personally identifiable information (PII)
- Data is aggregated and anonymized
- Sensitive attributes (race, gender) are included for fairness analysis only

## Citation

If using this dataset, please cite:

```
Dua, D. and Graff, C. (2019). UCI Machine Learning Repository 
[http://archive.ics.uci.edu/ml]. Irvine, CA: University of California, 
School of Information and Computer Science.
```

## Contact

For questions about data access or preprocessing, please open an issue in the repository.