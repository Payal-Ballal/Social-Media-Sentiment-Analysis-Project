
import re
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()


def clean_text(text):
    """Clean a social media post before analysis."""
    text = str(text)
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = text.replace("#", "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def analyze_sentiment(text):
    """Classify a post as Positive, Negative, or Neutral."""
    scores = analyzer.polarity_scores(text)
    compound = scores["compound"]

    if compound >= 0.05:
        sentiment = "Positive"
    elif compound <= -0.05:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"

    return sentiment, compound


def process_data(df):
    """Clean posts and add sentiment scores."""
    df = df.copy()
    df["text"] = df["text"].fillna("").apply(clean_text)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df = df[df["text"] != ""]

    results = df["text"].apply(analyze_sentiment)
    df["sentiment"] = results.apply(lambda result: result[0])
    df["compound_score"] = results.apply(lambda result: result[1])

    return df
