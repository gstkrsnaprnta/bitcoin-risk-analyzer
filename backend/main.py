from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
import requests
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import os
from pydantic import BaseModel

from news_scraper import get_recent_bitcoin_sentiment

app = FastAPI(title="Bitcoin Market Risk Analyzer API")

# Setup CORS for Frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # For dev, update this in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and scaler
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR) # Go up one level to 'Bitcoin'

try:
    model = joblib.load(os.path.join(ROOT_DIR, 'random_forest_bitcoin_model.pkl'))
    scaler = joblib.load(os.path.join(ROOT_DIR, 'scaler_bitcoin.pkl'))
except FileNotFoundError:
    model = None
    scaler = None

from cachetools import cached, TTLCache

# Cache data up to 15 seconds to simulate real-time without hitting rate limits
live_data_cache = TTLCache(maxsize=1, ttl=15)
fng_cache = TTLCache(maxsize=1, ttl=60)

@cached(cache=live_data_cache)
def get_live_data():
    """Mengambil data harga BTC 120 hari terakhir untuk Technical Indicators."""
    df = yf.download("BTC-USD", period='120d', interval='1d', auto_adjust=True)
    if df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df['price_change'] = df['Close'].pct_change()
    
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    df['volatility_7d'] = df['price_change'].rolling(window=7).std()
    
    return df.dropna().copy()

@cached(cache=fng_cache)
def get_fear_and_greed_score():
    try:
        res = requests.get('https://api.alternative.me/fng/?limit=1').json()
        return float(res['data'][0]['value'])
    except Exception:
        return 50.0

@app.get("/")
def read_root():
    return {"message": "Bitcoin Risk Analyzer API is running!"}

@app.get("/api/risk-index")
async def get_risk_index():
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model ML belum ditraining atau tidak ditemukan (model.pkl / scaler.pkl). Jalankan train.py.")

    try:
        # 1. Dapatkan live data BTC (RSI, Volatility, Price Change Lag1)
        data = get_live_data()
        if data.empty or len(data) < 5:
            raise HTTPException(status_code=500, detail="Gagal mengambil data dari Yahoo Finance")
            
        current_row = data.iloc[-1]
        prev_row = data.iloc[-2]
        
        current_close = float(current_row['Close'])
        current_rsi = float(current_row['rsi'])
        current_vol = float(current_row['volatility_7d'])
        prev_change = float(prev_row['price_change'])
        
        # 2. Fear and greed
        fng_score = get_fear_and_greed_score()
        
        # 3. Sentiment berita
        sentiment_sma3, sentiment_lag1, top_news = get_recent_bitcoin_sentiment()
        
        # Fitur array order: ['fng_score', 'rsi', 'volatility_7d', 'sentiment_sma3', 'price_change_lag1', 'sentiment_lag1']
        features_dict = {
            'fng_score': fng_score,
            'rsi': current_rsi,
            'volatility_7d': current_vol,
            'sentiment_sma3': sentiment_sma3,
            'price_change_lag1': prev_change,
            'sentiment_lag1': sentiment_lag1
        }
        
        df_input = pd.DataFrame([features_dict])
        
        # Scaling
        scaled_input = scaler.transform(df_input)
        
        # Prediksi
        raw_prediction = model.predict(scaled_input)[0]
        prediction = str(raw_prediction)
        
        # Determine actionable advice and colors based on the models output
        if prediction == "Low Risk":
            risk_score = 25
            color = "#10B981" # Green
            advice_title = "DCA Opportunity"
            advice_text = "The market is currently showing signs of being oversold while fundamentals remain stable. Historically, this is an ideal zone to accumulate (Dollar Cost Average) before potential reversals."
        elif prediction == "Medium Risk":
            risk_score = 50
            color = "#F59E0B" # Yellow
            advice_title = "Proceed with Caution"
            advice_text = "The market is in a transition phase or consolidating. Avoid large lump-sum purchases and monitor key technicals for the next breakout direction."
        else: # High Risk
            risk_score = 85
            color = "#EF4444" # Red
            advice_title = "Wait & See / Take Profits"
            advice_text = "Market volatility is extremely high and sentiment is deteriorating. Avoid FOMO buying as the risk of a sharp correction is significant. Consider taking profits if over-allocated."
            
        return {
            "timestamp": datetime.now().isoformat(),
            "btc_price": current_close,
            "prediction": prediction,
            "risk_score": risk_score,
            "color": color,
            "advice": {
                "title": advice_title,
                "text": advice_text
            },
            "top_news": top_news,
            "features": features_dict
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

class CustomFeatures(BaseModel):
    fng_score: float
    rsi: float
    volatility_7d: float
    sentiment_sma3: float
    price_change_lag1: float
    sentiment_lag1: float

@app.post("/api/predict-custom")
async def predict_custom_risk(features: CustomFeatures):
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model ML belum ditraining.")

    try:
        # Construct dataframe from user input
        features_dict = features.model_dump()
        df_input = pd.DataFrame([features_dict])
        
        # Scale
        scaled_input = scaler.transform(df_input)
        
        # Predict
        raw_prediction = model.predict(scaled_input)[0]
        prediction = str(raw_prediction)
        
        # Determine actionable advice and colors based on output
        if prediction == "Low Risk":
            risk_score = 25
            color = "#10B981" # Green
            advice_title = "DCA Opportunity"
            advice_text = "The market is currently showing signs of being oversold while fundamentals remain stable. Historically, this is an ideal zone to accumulate (Dollar Cost Average) before potential reversals."
        elif prediction == "Medium Risk":
            risk_score = 50
            color = "#F59E0B" # Yellow
            advice_title = "Proceed with Caution"
            advice_text = "The market is in a transition phase or consolidating. Avoid large lump-sum purchases and monitor key technicals for the next breakout direction."
        else: # High Risk
            risk_score = 85
            color = "#EF4444" # Red
            advice_title = "Wait & See / Take Profits"
            advice_text = "Market volatility is extremely high and sentiment is deteriorating. Avoid FOMO buying as the risk of a sharp correction is significant. Consider taking profits if over-allocated."
            
        return {
            "prediction": prediction,
            "risk_score": risk_score,
            "color": color,
            "advice": {
                "title": advice_title,
                "text": advice_text
            },
            "features": features_dict
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/history")
async def get_historical_data():
    """Endpoint untuk memberikan data riwayat harga, RSI, dan volatilitas untuk Recharts di Frontend."""
    try:
        # Ambil data live (120 hari) lalu ambil 30 hari terakhir untuk plotting
        data = get_live_data()
        
        if data.empty:
            raise HTTPException(status_code=500, detail="Gagal mengambil data riwayat.")
            
        recent_data = data.tail(30).copy()
        
        # Format ke JSON untuk Recharts
        history_list = []
        for index, row in recent_data.iterrows():
            history_list.append({
                "date": index.strftime('%Y-%m-%d'),
                "price": round(float(row['Close']), 2),
                "rsi": round(float(row['rsi']), 2),
                "volatility": round(float(row['volatility_7d']) * 100, 2) # Format persentase
            })
            
        return {
            "history": history_list
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
