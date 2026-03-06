import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_and_preprocess_data(csv_path):
    """
    Loads German Credit data from CSV and performs basic preprocessing.
    Returns X (features) and y (target).
    """

    df = pd.read_csv(csv_path)

    print(f"✓ Loaded dataset from local CSV: {csv_path}")
    print(f"Shape: {df.shape}, Default rate: {df['target'].mean():.1%}")

    # Separate target
    y = df["target"]
    X = df.drop(columns=["target"])

    # Encode categorical variables
    categorical_cols = X.select_dtypes(include=["object"]).columns

    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

    print(f"✓ Preprocessing complete: X={X.shape}, y={y.shape}")

    return X, y
