"""
================================================================================
Exploratory Data Analysis (EDA) Module
================================================================================

Functions for analyzing and visualizing IMDb sentiment data.

Includes:
- Word cloud generation
- Sentiment distribution visualization
- Review length analysis
- Frequent words analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
import os

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ============================================================================
# MAIN VISUALIZATION FUNCTION
# ============================================================================

def generate_visualizations(df, output_dir='outputs/visualizations/'):
    """
    Generate all visualizations for EDA.
    
    Args:
        df (pd.DataFrame): Dataset with 'review' and 'sentiment' columns
        output_dir (str): Directory to save visualizations
    """
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate each visualization
    create_word_clouds(df, output_dir)
    create_sentiment_distribution(df, output_dir)
    create_review_length_distribution(df, output_dir)
    create_most_frequent_words(df, output_dir)

# ============================================================================
# WORD CLOUD FUNCTIONS
# ============================================================================

def create_word_clouds(df, output_dir='outputs/visualizations/'):
    """
    Create word clouds for positive and negative reviews.
    
    Args:
        df (pd.DataFrame): Dataset with 'review' and 'sentiment' columns
        output_dir (str): Directory to save images
    """
    
    # Separate reviews by sentiment
    positive_reviews = df[df['sentiment'] == 'positive']['review'].str.lower()
    negative_reviews = df[df['sentiment'] == 'negative']['review'].str.lower()
    
    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Positive word cloud
    positive_text = ' '.join(positive_reviews)
    positive_wc = WordCloud(
        width=800,
        height=400,
        background_color='white',
        colormap='Greens',
        max_words=100
    ).generate(positive_text)
    
    axes[0].imshow(positive_wc, interpolation='bilinear')
    axes[0].set_title('Top Words in Positive Reviews', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # Negative word cloud
    negative_text = ' '.join(negative_reviews)
    negative_wc = WordCloud(
        width=800,
        height=400,
        background_color='white',
        colormap='Reds',
        max_words=100
    ).generate(negative_text)
    
    axes[1].imshow(negative_wc, interpolation='bilinear')
    axes[1].set_title('Top Words in Negative Reviews', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'word_clouds.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Word clouds saved to: {output_dir}word_clouds.png")

# ============================================================================
# SENTIMENT DISTRIBUTION FUNCTIONS
# ============================================================================

def create_sentiment_distribution(df, output_dir='outputs/visualizations/'):
    """
    Create sentiment distribution visualization.
    
    Args:
        df (pd.DataFrame): Dataset with 'sentiment' column
        output_dir (str): Directory to save image
    """
    
    sentiment_counts = df['sentiment'].value_counts()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Pie chart
    colors = ['#2ecc71', '#e74c3c']  # Green for positive, red for negative
    axes[0].pie(
        sentiment_counts.values,
        labels=sentiment_counts.index,
        autopct='%1.1f%%',
        colors=colors,
        startangle=90,
        textprops={'fontsize': 12, 'fontweight': 'bold'}
    )
    axes[0].set_title('Sentiment Distribution (Pie Chart)', fontsize=14, fontweight='bold')
    
    # Bar chart
    sentiment_counts.plot(
        kind='bar',
        ax=axes[1],
        color=colors,
        edgecolor='black',
        linewidth=1.5
    )
    axes[1].set_title('Sentiment Distribution (Bar Chart)', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Sentiment', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Count', fontsize=12, fontweight='bold')
    axes[1].tick_params(axis='x', rotation=0)
    
    # Add count labels on bars
    for i, v in enumerate(sentiment_counts.values):
        axes[1].text(i, v + 500, str(v), ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sentiment_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Sentiment distribution saved to: {output_dir}sentiment_distribution.png")

# ============================================================================
# REVIEW LENGTH FUNCTIONS
# ============================================================================

def create_review_length_distribution(df, output_dir='outputs/visualizations/'):
    """
    Create review length distribution visualization.
    
    Args:
        df (pd.DataFrame): Dataset with 'review' column
        output_dir (str): Directory to save image
    """
    
    # Calculate review lengths
    df['review_length'] = df['review'].apply(lambda x: len(str(x).split()))
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    axes[0].hist(
        df['review_length'],
        bins=50,
        color='skyblue',
        edgecolor='black',
        alpha=0.7
    )
    axes[0].set_title('Review Length Distribution', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Number of Words', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Frequency', fontsize=12, fontweight='bold')
    axes[0].axvline(df['review_length'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {df["review_length"].mean():.0f}')
    axes[0].axvline(df['review_length'].median(), color='green', linestyle='--', linewidth=2, label=f'Median: {df["review_length"].median():.0f}')
    axes[0].legend()
    
    # Box plot by sentiment
    sentiment_groups = [df[df['sentiment'] == 'positive']['review_length'], 
                       df[df['sentiment'] == 'negative']['review_length']]
    
    axes[1].boxplot(
        sentiment_groups,
        labels=['Positive', 'Negative'],
        patch_artist=True,
        boxprops=dict(facecolor='lightblue'),
        medianprops=dict(color='red', linewidth=2)
    )
    axes[1].set_title('Review Length by Sentiment', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Number of Words', fontsize=12, fontweight='bold')
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'review_length_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Review length distribution saved to: {output_dir}review_length_distribution.png")

# ============================================================================
# FREQUENT WORDS FUNCTIONS
# ============================================================================

def create_most_frequent_words(df, output_dir='outputs/visualizations/', top_n=15):
    """
    Create visualization of most frequent words by sentiment.
    
    Args:
        df (pd.DataFrame): Dataset with 'review' and 'sentiment' columns
        output_dir (str): Directory to save image
        top_n (int): Number of top words to display
    """
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Positive reviews
    positive_words = ' '.join(df[df['sentiment'] == 'positive']['review']).lower().split()
    positive_freq = Counter(positive_words).most_common(top_n)
    pos_words, pos_counts = zip(*positive_freq)
    
    axes[0].barh(pos_words, pos_counts, color='#2ecc71', edgecolor='black')
    axes[0].set_title(f'Top {top_n} Most Frequent Words - Positive Reviews', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Frequency', fontsize=11, fontweight='bold')
    axes[0].invert_yaxis()
    axes[0].grid(axis='x', alpha=0.3)
    
    # Negative reviews
    negative_words = ' '.join(df[df['sentiment'] == 'negative']['review']).lower().split()
    negative_freq = Counter(negative_words).most_common(top_n)
    neg_words, neg_counts = zip(*negative_freq)
    
    axes[1].barh(neg_words, neg_counts, color='#e74c3c', edgecolor='black')
    axes[1].set_title(f'Top {top_n} Most Frequent Words - Negative Reviews', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Frequency', fontsize=11, fontweight='bold')
    axes[1].invert_yaxis()
    axes[1].grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'frequent_words.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Frequent words visualization saved to: {output_dir}frequent_words.png")

# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def analyze_sentiment_distribution(df):
    """
    Analyze and print sentiment distribution statistics.
    
    Args:
        df (pd.DataFrame): Dataset with 'sentiment' column
    """
    
    print("\n" + "="*60)
    print("SENTIMENT DISTRIBUTION ANALYSIS")
    print("="*60)
    
    sentiment_counts = df['sentiment'].value_counts()
    
    for sentiment, count in sentiment_counts.items():
        percentage = (count / len(df)) * 100
        print(f"{sentiment.capitalize():12} : {count:6,} ({percentage:5.2f}%)")
    
    print("="*60)
    
    # Check balance
    counts = sentiment_counts.values
    imbalance_ratio = max(counts) / min(counts)
    
    if imbalance_ratio < 1.1:
        print("✓ Dataset is well-balanced")
    elif imbalance_ratio < 1.5:
        print("⚠ Dataset is slightly imbalanced")
    else:
        print("❌ Dataset is significantly imbalanced")

def get_text_statistics(df, text_column='review'):
    """
    Get text statistics.
    
    Args:
        df (pd.DataFrame): Dataset
        text_column (str): Name of text column
    
    Returns:
        dict: Dictionary of statistics
    """
    
    df['word_count'] = df[text_column].apply(lambda x: len(str(x).split()))
    df['char_count'] = df[text_column].apply(lambda x: len(str(x)))
    
    stats = {
        'avg_words': df['word_count'].mean(),
        'median_words': df['word_count'].median(),
        'min_words': df['word_count'].min(),
        'max_words': df['word_count'].max(),
        'avg_chars': df['char_count'].mean(),
        'median_chars': df['char_count'].median(),
    }
    
    return stats