import json
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Load the JSON data
with open('/Users/fatihozkan/Desktop/Projectttt/finrl-deepseek-stock-prediction/sentiment/temp/processed_data.json', 'r') as f:
    data = json.load(f)

# Initialize the sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Extract datetime and content, then compute sentiment scores
sentiment_data = []
for entry in data:
    datetime = entry['datetime']
    content = entry['content']
    sentiment_score = analyzer.polarity_scores(content)['compound']
    sentiment_data.append({'datetime': datetime, 'sentiment_score': sentiment_score})

# Convert to DataFrame and save as CSV
df = pd.DataFrame(sentiment_data)
df.to_csv('sentiment/sentiment_scores.csv', index=False)

print("Sentiment scores saved to sentiment/sentiment_scores.csv") 