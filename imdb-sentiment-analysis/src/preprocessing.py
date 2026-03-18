"""
================================================================================
Text Preprocessing Module
================================================================================

Functions for cleaning and preparing text data for sentiment analysis.

Includes:
- HTML tag removal
- Punctuation & number removal
- Tokenization
- Stopword removal
- Lemmatization
"""

import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import html

# Download required NLTK data (run once)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Get English stopwords
stop_words = set(stopwords.words('english'))

# ============================================================================
# MAIN PREPROCESSING FUNCTION
# ============================================================================

def preprocess_text(text):
    """
    Complete preprocessing pipeline for a single review.
    
    Steps:
    1. Remove HTML tags
    2. Convert to lowercase
    3. Remove URLs
    4. Remove special characters and punctuation
    5. Tokenize
    6. Remove stopwords
    7. Lemmatize
    
    Args:
        text (str): Raw review text
    
    Returns:
        str: Cleaned and preprocessed text
    
    Example:
        >>> review = "<br/>This movie was AMAZING! 5/5 stars!!!"
        >>> preprocess_text(review)
        'movie amazing'
    """
    
    if not isinstance(text, str):
        return ""
    
    # Step 1: Remove HTML tags
    text = remove_html_tags(text)
    
    # Step 2: Convert to lowercase
    text = text.lower()
    
    # Step 3: Remove URLs
    text = remove_urls(text)
    
    # Step 4: Remove special characters, keep only alphanumeric and spaces
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    
    # Step 5: Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()
    
    # Step 6: Tokenize
    tokens = word_tokenize(text)
    
    # Step 7: Remove stopwords and lemmatize
    cleaned_tokens = []
    for token in tokens:
        # Skip stopwords and very short tokens
        if token not in stop_words and len(token) > 2:
            # Lemmatize
            lemmatized = lemmatizer.lemmatize(token)
            cleaned_tokens.append(lemmatized)
    
    # Step 8: Join tokens back to string
    cleaned_text = " ".join(cleaned_tokens)
    
    return cleaned_text

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def remove_html_tags(text):
    """
    Remove HTML tags from text.
    
    Args:
        text (str): Text with HTML tags
    
    Returns:
        str: Text without HTML tags
    """
    # Remove <br> and <br/> tags
    text = re.sub(r"<br\s*/?>\s*", " ", text)
    
    # Remove all other HTML tags
    text = re.sub(r"<[^>]+>", "", text)
    
    # Decode HTML entities
    text = html.unescape(text)
    
    return text

def remove_urls(text):
    """
    Remove URLs from text.
    
    Args:
        text (str): Text containing URLs
    
    Returns:
        str: Text without URLs
    """
    # Remove http, https, ftp URLs
    text = re.sub(r"http\S+|https\S+|ftp\S+", "", text)
    
    # Remove www URLs
    text = re.sub(r"www\.\S+", "", text)
    
    return text

def remove_contractions(text):
    """
    Expand contractions (e.g., "don't" → "do not").
    
    Args:
        text (str): Text with contractions
    
    Returns:
        str: Text with expanded contractions
    """
    contractions_dict = {
        "ain't": "am not",
        "aren't": "are not",
        "can't": "cannot",
        "could've": "could have",
        "couldn't": "could not",
        "didn't": "did not",
        "doesn't": "does not",
        "don't": "do not",
        "hadn't": "had not",
        "hasn't": "has not",
        "haven't": "have not",
        "he'd": "he would",
        "he'll": "he will",
        "he's": "he is",
        "i'd": "i would",
        "i'll": "i will",
        "i'm": "i am",
        "i've": "i have",
        "isn't": "is not",
        "it's": "it is",
        "let's": "let us",
        "shouldn't": "should not",
        "that's": "that is",
        "there's": "there is",
        "they'd": "they would",
        "they'll": "they will",
        "they're": "they are",
        "they've": "they have",
        "wasn't": "was not",
        "we'd": "we would",
        "we'll": "we will",
        "we're": "we are",
        "we've": "we have",
        "weren't": "were not",
        "what's": "what is",
        "won't": "will not",
        "wouldn't": "would not",
        "you'd": "you would",
        "you'll": "you will",
        "you're": "you are",
        "you've": "you have"
    }
    
    for contraction, expansion in contractions_dict.items():
        text = re.sub(r'\b' + contraction + r'\b', expansion, text.lower())
    
    return text

def remove_emojis(text):
    """Remove emojis from text."""
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"
        "\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF"
        "\U0001F1E0-\U0001F1FF"
        "\U0001F900-\U0001F9FF"
        "]+", flags=re.UNICODE
    )
    return emoji_pattern.sub(r'', text)

def clean_dataset(df, text_column='review'):
    """
    Clean entire dataset.
    
    Args:
        df (pd.DataFrame): DataFrame with text column
        text_column (str): Name of column containing text
    
    Returns:
        pd.DataFrame: DataFrame with cleaned text
    """
    df_copy = df.copy()
    df_copy[text_column] = df_copy[text_column].apply(preprocess_text)
    return df_copy