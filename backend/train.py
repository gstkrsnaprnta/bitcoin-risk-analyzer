import pandas as pd
import numpy as np
import yfinance as yf
import requests
import joblib
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import os

def train():
    print("Mulai mengambil data...")
    # 1. Download Data Harga BTC
    df_raw = yf.download("BTC-USD", start="2021-11-05", interval="1d")
    df_price = df_raw.copy()
    if isinstance(df_price.columns, pd.MultiIndex):
        df_price.columns = df_price.columns.get_level_values(0)
    df_price = df_price.reset_index()
    df_price['Date'] = pd.to_datetime(df_price['Date']).dt.tz_localize(None).dt.normalize()
    df_price = df_price[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
    df_price = df_price.drop_duplicates(subset='Date').sort_values('Date').reset_index(drop=True)

    # 2. Load Data News (CSV)
    df_news = pd.read_csv('../bitcoin_news_2021_2024.csv')
    df_news_clean = df_news.drop_duplicates().copy()
    df_news_clean['Date'] = pd.to_datetime(df_news_clean['Date']).dt.normalize()
    df_news_daily = df_news_clean.groupby('Date').agg({
        'Accurate Sentiments': 'mean',
        'Short Description': 'count'
    }).reset_index()
    df_news_daily.columns = ['Date', 'avg_sentiment', 'news_count']

    # 3. Load Data Fear & Greed Index
    fng_url = "https://api.alternative.me/fng/?limit=0&format=json"
    response = requests.get(fng_url)
    fng_data = response.json()['data']
    df_fng = pd.DataFrame(fng_data)
    df_fng['timestamp'] = pd.to_numeric(df_fng['timestamp'])
    df_fng['timestamp'] = pd.to_datetime(df_fng['timestamp'], unit='s')
    df_fng.rename(columns={'timestamp': 'Date', 'value': 'fng_score'}, inplace=True)
    df_fng['fng_score'] = pd.to_numeric(df_fng['fng_score'], errors='coerce')
    df_fng_clean = df_fng[['Date', 'fng_score']].copy()
    df_fng_clean['Date'] = pd.to_datetime(df_fng_clean['Date']).dt.normalize()
    df_fng_clean = df_fng_clean.drop_duplicates(subset='Date')

    # 4. Merging
    df_merged_1 = pd.merge(df_price, df_fng_clean, on='Date', how='inner')
    df_master = pd.merge(df_merged_1, df_news_daily, on='Date', how='inner')
    df_master = df_master.sort_values('Date').reset_index(drop=True)

    # 5. Feature Engineering
    df_master['price_change'] = df_master['Close'].pct_change()
    delta = df_master['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df_master['rsi'] = 100 - (100 / (1 + rs))
    df_master['volatility_7d'] = df_master['price_change'].rolling(window=7).std()
    df_master['sentiment_sma3'] = df_master['avg_sentiment'].rolling(window=3).mean()
    
    df_features = df_master.dropna().copy()
    df_features['price_change_lag1'] = df_features['price_change'].shift(1)
    df_features['sentiment_lag1'] = df_features['avg_sentiment'].shift(1)
    df_final = df_features.dropna().copy()

    final_features_list = [
        'fng_score', 'rsi', 'volatility_7d', 'sentiment_sma3', 'price_change_lag1', 'sentiment_lag1'
    ]

    scaler_final = StandardScaler()
    scaled_matrix = scaler_final.fit_transform(df_final[final_features_list])
    df_modeling = pd.DataFrame(scaled_matrix, columns=final_features_list, index=df_final.index)

    # 6. KMeans Labeling
    kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=300, n_init=10, random_state=42)
    clusters = kmeans.fit_predict(df_modeling)
    df_final['Cluster'] = clusters
    
    cluster_profile = df_final.groupby('Cluster').agg({'volatility_7d': 'mean'}).sort_values(by='volatility_7d')
    risk_mapping = {
        cluster_profile.index[0]: 'Low Risk',
        cluster_profile.index[1]: 'Medium Risk',
        cluster_profile.index[2]: 'High Risk'
    }
    df_final['Risk_Level'] = df_final['Cluster'].map(risk_mapping)

    # 7. Random Forest Training
    X = df_modeling
    y = df_final['Risk_Level']
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X, y)

    # Save
    joblib.dump(rf_model, 'model.pkl')
    joblib.dump(scaler_final, 'scaler.pkl')
    df_final.to_csv('bitcoin_final_data.csv', index=False)
    print("Training Selesai! Model dan Scaler disimpan ke backend/model.pkl dan backend/scaler.pkl")

if __name__ == "__main__":
    train()
