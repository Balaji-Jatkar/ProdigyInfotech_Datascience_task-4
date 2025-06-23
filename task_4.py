import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
from wordcloud import WordCloud
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('D:\\Internship projects\\Internship 1\\task_4\\sentimentdataset.csv')

print("Dataset shape:", df.shape)
print("\nFirst few rows:")
print(df.head())

print("\nDataset info:")
print(df.info())

print("\nMissing values:")
print(df.isnull().sum())

df = df.dropna()

if len(df) > 5000:
    df = df.sample(n=5000, random_state=42)
    print(f"Sampled dataset to 5000 rows for faster processing")

def get_sentiment(text):
    blob = TextBlob(str(text))
    polarity = blob.sentiment.polarity
    if polarity > 0:
        return 'Positive'
    elif polarity < 0:
        return 'Negative'
    else:
        return 'Neutral'

df['sentiment'] = df['Text'].apply(get_sentiment)

print("\nSentiment Distribution:")
print(df['sentiment'].value_counts())

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sentiment_counts = df['sentiment'].value_counts()
colors = ['#2ecc71', '#e74c3c', '#95a5a6']  # green, red, gray
plt.pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%', colors=colors)
plt.title('Overall Sentiment Distribution')

plt.subplot(1, 2, 2)
ax = sns.countplot(data=df, x='sentiment', palette=['#2ecc71', '#e74c3c', '#95a5a6'])
plt.title('Sentiment Counts')
plt.ylabel('Number of Posts')

for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width()/2., p.get_height()), 
                ha='center', va='bottom')

plt.tight_layout()
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sentiments = ['Positive', 'Negative']
colors = ['Greens', 'Reds']

for i, sentiment in enumerate(sentiments):
    text = ' '.join(df[df['sentiment'] == sentiment]['Text'].astype(str))
    if text.strip():
        wordcloud = WordCloud(width=500, height=300, 
                            background_color='white',
                            colormap=colors[i],
                            max_words=50).generate(text)
        axes[i].imshow(wordcloud, interpolation='bilinear')
        axes[i].set_title(f'{sentiment} Words', fontsize=14, fontweight='bold')
        axes[i].axis('off')

plt.tight_layout()
plt.show()

print("\n=== ANALYSIS SUMMARY ===")
print(f"Total posts analyzed: {len(df)}")
print(f"Positive sentiment: {(df['sentiment'] == 'Positive').sum()} ({(df['sentiment'] == 'Positive').mean()*100:.1f}%)")
print(f"Negative sentiment: {(df['sentiment'] == 'Negative').sum()} ({(df['sentiment'] == 'Negative').mean()*100:.1f}%)")
print(f"Neutral sentiment: {(df['sentiment'] == 'Neutral').sum()} ({(df['sentiment'] == 'Neutral').mean()*100:.1f}%)")

from collections import Counter

def get_common_words(text_series, n=10):
    words = ' '.join(text_series.astype(str)).lower().split()
    stop_words = ['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should']
    words = [word for word in words if word not in stop_words and len(word) > 2]
    return Counter(words).most_common(n)

print("\nTop 5 words in POSITIVE posts:")
pos_words = get_common_words(df[df['sentiment'] == 'Positive']['Text'], n=5)
for word, count in pos_words:
    print(f"  {word}: {count}")

print("\nTop 5 words in NEGATIVE posts:")
neg_words = get_common_words(df[df['sentiment'] == 'Negative']['Text'], n=5)
for word, count in neg_words:
    print(f"  {word}: {count}")

df.to_csv('sentiment_analysis_results.csv', index=False)
print("\nResults saved to 'sentiment_analysis_results.csv'")