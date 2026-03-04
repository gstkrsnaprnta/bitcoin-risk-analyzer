import feedparser
from bs4 import BeautifulSoup
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
from datetime import datetime, timedelta

# Pastikan lexicon di-download
try:
    nltk.data.find('sentiment/vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

from cachetools import cached, TTLCache

# Cache news scraping up to 60 seconds to avoid IP bans from Google News RSS
news_cache = TTLCache(maxsize=1, ttl=60)

@cached(cache=news_cache)
def get_recent_bitcoin_sentiment():
    """
    Mengambil berita Bitcoin dari RSS Google News,
    dan menghitung rata-rata sentimen (3 hari terakhir).
    """
    rss_url = "https://news.google.com/rss/search?q=Bitcoin&hl=en-US&gl=US&ceid=US:en"
    feed = feedparser.parse(rss_url)
    
    sia = SentimentIntensityAnalyzer()
    
    news_data = []
    
    for entry in feed.entries:
        try:
            # Parse judul berita dan URL
            title = entry.title
            link = entry.link
            
            # Waktu publikasi
            pub_date = pd.to_datetime(entry.published).tz_localize(None).normalize()
            
            # Hitung Sentimen
            score = sia.polarity_scores(title)['compound']
            
            news_data.append({
                'Date': pub_date,
                'title': title,
                'link': link,
                'sentiment': score
            })
        except Exception as e:
            continue
            
    df = pd.DataFrame(news_data)
    
    if df.empty:
        # Jika gagal atau kosong, kembalikan default
        return 0.0, 0.0
        
    # Group by Date
    daily_sentiment = df.groupby('Date')['sentiment'].mean().reset_index()
    daily_sentiment = daily_sentiment.sort_values('Date').reset_index(drop=True)
    
    # Ambil data hari ini, kemarin, dst.
    today = pd.Timestamp.today().normalize()
    
    # Fill missing dates with 0 sentiment or forward fill
    dates_to_check = [today - timedelta(days=i) for i in range(4)]
    
    sentima_dict = dict(zip(daily_sentiment['Date'], daily_sentiment['sentiment']))
    
    s_today = sentima_dict.get(today, 0.0)
    s_lag1 = sentima_dict.get(today - timedelta(days=1), s_today)
    s_lag2 = sentima_dict.get(today - timedelta(days=2), s_lag1)
    s_lag3 = sentima_dict.get(today - timedelta(days=3), s_lag2)
    
    # sentiment_sma3 = rata-rata (today, lag1, lag2)
    sentiment_sma3 = (s_today + s_lag1 + s_lag2) / 3
    
    # Ambil berita teratas (maksimal 5 berita yang baru dipublish)
    top_news = []
    # Urutkan berita berdasarkan tanggal terbaru dan ambil 5 teratas
    recent_news = df.sort_values(by='Date', ascending=False).head(5)
    for _, row in recent_news.iterrows():
        top_news.append({
            "title": row['title'],
            "link": row['link'],
            "sentiment": row['sentiment'],
            "date": row['Date'].isoformat()
        })
    
    return sentiment_sma3, s_lag1, top_news

if __name__ == "__main__":
    sma3, lag1, top = get_recent_bitcoin_sentiment()
    print(f"Sentiment SMA3: {sma3}, Sentiment Lag1: {lag1}")
