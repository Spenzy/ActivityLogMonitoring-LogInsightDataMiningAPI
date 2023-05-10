import nltk
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer

nltk.download('vader_lexicon')

# Initialize the sentiment analyzer
sia = SentimentIntensityAnalyzer()


# Load the data
df = pd.read_csv('DataMining/GartnerData.csv', sep='|', lineterminator='\n', header=0)
df = df.drop_duplicates()

# Compute the sentiment scores for each row in the Text column
df['Sentiment'] = df['Text'].apply(lambda x: sia.polarity_scores(x)['compound'])

# Compute descriptive statistics for the sentiment scores
stats = df['Sentiment'].describe()

# Print the descriptive statistics
print(stats)

# Create a new column with sentiment categories
df['Sentiment_Category'] = pd.cut(df['Sentiment'], bins=[-1, -0.33, 0.33, 1], labels=['Negative', 'Neutral', 'Positive'])

def main():
    return df