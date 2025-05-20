import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from textblob import TextBlob
import json

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

df = pd.read_csv("workspace/csv_files/comments-list_comments_captured-list_2025-05-19_17-36-47_5c07cdcf-f49f-48ac-a5e2-9339fa2df228.csv")

processed_comments = []
sentiment_counts = {"positive": 0, "neutral": 0, "negative": 0}

for index, row in df.iterrows():
    text = str(row["Comment Text"])
    tokens = word_tokenize(text)
    
    filtered = [word for word in tokens if word.lower() not in stop_words and word.isalpha()]
    
    stemmed = [stemmer.stem(word) for word in filtered]
    
    lemmatized = [lemmatizer.lemmatize(word) for word in filtered]
    
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    if polarity > 0.1:
        sentiment = "positive"
    elif polarity < -0.1:
        sentiment = "negative"
    else:
        sentiment = "neutral"
    sentiment_counts[sentiment] += 1

    processed_comments.append({
        "original": text,
        "tokens": tokens,
        "filtered": filtered,
        "stemmed": stemmed,
        "lemmatized": lemmatized,
        "sentiment": sentiment
    })

with open("processed_comments.json", "w", encoding="utf-8") as f:
    json.dump(processed_comments, f, ensure_ascii=False, indent=4)

print("Sentiment analysis results:")
print(sentiment_counts)
