from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
import requests
import pickle
import json
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
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Load new Gradient Boosting model & config ──────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

try:
    with open(os.path.join(BASE_DIR, 'btc_risk_model.pkl'), 'rb') as f:
        model = pickle.load(f)
    with open(os.path.join(BASE_DIR, 'model_threshold.json'), 'r') as f:
        config = json.load(f)
    with open(os.path.join(BASE_DIR, 'feature_stats.json'), 'r') as f:
        feature_stats = json.load(f)
    THRESHOLD = config['threshold']       # 0.59
    FEAT_COLS = config['feature_cols']     # list of 12 features
except FileNotFoundError as e:
    print(f"[WARNING] Model files not found: {e}")
    model = None
    THRESHOLD = 0.59
    FEAT_COLS = []
    feature_stats = {}

# ─── Cache setup ─────────────────────────────────────────────────────────────
from cachetools import cached, TTLCache

live_data_cache = TTLCache(maxsize=1, ttl=15)
fng_cache = TTLCache(maxsize=1, ttl=60)


@cached(cache=live_data_cache)
def get_live_data():
    """Fetch 120 days of BTC price data for technical indicators."""
    df = yf.download("BTC-USD", period='120d', interval='1d', auto_adjust=True)
    if df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    return df.copy()


@cached(cache=fng_cache)
def get_fear_and_greed_score():
    try:
        res = requests.get('https://api.alternative.me/fng/?limit=1').json()
        return float(res['data'][0]['value'])
    except Exception:
        return 50.0


# ─── Feature computation ────────────────────────────────────────────────────
def compute_features(df_price, fng_score, mean_sentiment, news_count):
    """Compute 12 model features + RSI & volatility for display."""
    close  = df_price['Close']
    high   = df_price['High']
    low    = df_price['Low']
    volume = df_price['Volume']

    daily_return   = close.pct_change()
    vol_7d         = daily_return.rolling(7).std()
    ma14           = close.rolling(14).mean()
    dist_to_ma14   = (close / ma14) - 1

    # RSI (for UI display, not a model feature)
    delta = close.diff()
    gain  = delta.where(delta > 0, 0).rolling(14).mean()
    loss  = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rsi   = 100 - (100 / (1 + gain / (loss + 1e-9)))

    # MACD
    ema12        = close.ewm(span=12, adjust=False).mean()
    ema26        = close.ewm(span=26, adjust=False).mean()
    macd_line    = ema12 - ema26
    macd_signal  = macd_line.ewm(span=9, adjust=False).mean()
    macd_hist    = macd_line - macd_signal
    macd_hist_norm = macd_hist / (close + 1e-9)

    # Bollinger Bands (window=20)
    bb_mid      = close.rolling(20).mean()
    bb_std      = close.rolling(20).std()
    bb_upper    = bb_mid + 2 * bb_std
    bb_lower    = bb_mid - 2 * bb_std
    bb_width    = (bb_upper - bb_lower) / (bb_mid + 1e-9)
    bb_position = (close - bb_lower) / (bb_upper - bb_lower + 1e-9)

    # HL Range & Volume MA Ratio
    hl_range        = (high - low) / (close + 1e-9)
    volume_ma_ratio = volume / (volume.rolling(14).mean() + 1e-9)

    # Sentiment-based interaction features
    sentiment_x_vol = mean_sentiment * float(vol_7d.iloc[-1])
    fear_pressure   = 1 if (fng_score < 30 and mean_sentiment < 0) else 0

    # Build feature dict (today's values)
    features = {
        'Daily_Return'          : float(daily_return.iloc[-1]),
        'Volume'                : float(volume.iloc[-1]),
        'Dist_to_MA14'          : float(dist_to_ma14.iloc[-1]),
        'fng_score'             : float(fng_score),
        'news_count'            : float(news_count),
        'Sentiment_x_Volatility': float(sentiment_x_vol),
        'Fear_Pressure'         : float(fear_pressure),
        'MACD_hist_norm'        : float(macd_hist_norm.iloc[-1]),
        'BB_Width'              : float(bb_width.iloc[-1]),
        'BB_Position'           : float(bb_position.iloc[-1]),
        'HL_Range'              : float(hl_range.iloc[-1]),
        'Volume_MA_Ratio'       : float(volume_ma_ratio.iloc[-1]),
    }

    # Extra display values
    rsi_value    = float(rsi.iloc[-1])
    vol_7d_value = float(vol_7d.iloc[-1])

    # MACD signal interpretation
    macd_hist_last = float(macd_hist.iloc[-1])
    if macd_hist_last > 0:
        macd_signal_label = "Bullish"
    elif macd_hist_last < 0:
        macd_signal_label = "Bearish"
    else:
        macd_signal_label = "Neutral"

    return features, rsi_value, vol_7d_value, macd_signal_label


# ─── Prediction helper ──────────────────────────────────────────────────────
def predict_risk(features_dict):
    """Run prediction using the Gradient Boosting model."""
    feat_values = [features_dict[col] for col in FEAT_COLS]
    feat_array  = np.array(feat_values).reshape(1, -1)
    prob_high   = float(model.predict_proba(feat_array)[0][1])
    prediction  = "High Risk" if prob_high >= THRESHOLD else "Normal"
    confidence  = round(
        prob_high * 100 if prob_high >= THRESHOLD else (1 - prob_high) * 100,
        1
    )

    if prediction == "High Risk":
        risk_score = round(prob_high * 100)
        color      = "#EF4444"
        advice     = {
            "title": "Wait & See",
            "text" : "High volatility detected with extreme RSI and bearish signals. "
                     "Avoid opening new positions. Consider securing profits or "
                     "waiting for market stabilization before re-entering."
        }
    else:
        risk_score = round((1 - prob_high) * 100)
        color      = "#10B981"
        advice     = {
            "title": "DCA Opportunity",
            "text" : "Market conditions appear stable with controlled volatility. "
                     "This is a historically favorable zone to accumulate Bitcoin "
                     "using Dollar Cost Averaging (DCA) strategy."
        }

    return prediction, risk_score, confidence, color, advice, prob_high


# ─── Endpoints ───────────────────────────────────────────────────────────────

@app.get("/")
def read_root():
    return {"message": "Bitcoin Risk Analyzer API is running!"}


@app.get("/api/risk-index")
async def get_risk_index():
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not found. Ensure btc_risk_model.pkl is in the backend/ folder."
        )

    try:
        # 1. Live BTC data
        data = get_live_data()
        if data.empty or len(data) < 30:
            raise HTTPException(status_code=500, detail="Failed to fetch data from Yahoo Finance")

        current_close = float(data['Close'].iloc[-1])

        # 2. Fear & Greed
        fng_score = get_fear_and_greed_score()

        # 3. News sentiment
        sentiment_sma3, sentiment_lag1, top_news = get_recent_bitcoin_sentiment()
        news_count = len(top_news) if top_news else 0

        # 4. Compute features
        features, rsi_value, vol_7d_value, macd_signal_label = compute_features(
            data, fng_score, sentiment_sma3, news_count
        )

        # 5. Predict
        prediction, risk_score, confidence, color, advice, prob_high = predict_risk(features)

        return {
            "timestamp"    : datetime.now().isoformat(),
            "btc_price"    : current_close,
            "prediction"   : prediction,
            "risk_score"   : risk_score,
            "confidence"   : confidence,
            "color"        : color,
            "advice"       : advice,
            "top_news"     : top_news,
            "fng_score"    : fng_score,
            "rsi"          : round(rsi_value, 2),
            "volatility_7d": round(vol_7d_value, 6),
            "bb_width"     : round(features['BB_Width'], 4),
            "macd_signal"  : macd_signal_label,
            "hl_range"     : round(features['HL_Range'], 4),
            "sentiment"    : round(sentiment_sma3, 4),
            "features"     : features,
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ─── Custom Prediction (Playground) ─────────────────────────────────────────

class PredictRequest(BaseModel):
    Daily_Return          : float = 0.0
    Volume                : float = 2.5e10
    Dist_to_MA14          : float = 0.0
    fng_score             : float = 50.0
    news_count            : float = 10.0
    Sentiment_x_Volatility: float = 0.0
    Fear_Pressure         : int   = 0
    MACD_hist_norm        : float = 0.0
    BB_Width              : float = 0.05
    BB_Position           : float = 0.5
    HL_Range              : float = 0.02
    Volume_MA_Ratio       : float = 1.0


@app.post("/api/predict-custom")
async def predict_custom_risk(req: PredictRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not found.")

    try:
        features_dict = req.model_dump()
        prediction, risk_score, confidence, color, advice, prob_high = predict_risk(features_dict)

        return {
            "prediction" : prediction,
            "risk_score"  : risk_score,
            "confidence"  : confidence,
            "color"       : color,
            "advice"      : advice,
            "features"    : features_dict,
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ─── Historical Data ────────────────────────────────────────────────────────

@app.get("/api/history")
async def get_historical_data():
    """Return 30 days of price, RSI, BB Width, and MACD histogram data."""
    try:
        data = get_live_data()
        if data.empty:
            raise HTTPException(status_code=500, detail="Failed to fetch historical data.")

        close  = data['Close']
        high   = data['High']
        low    = data['Low']
        volume = data['Volume']

        # RSI
        delta = close.diff()
        gain  = delta.where(delta > 0, 0).rolling(14).mean()
        loss  = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rsi   = 100 - (100 / (1 + gain / (loss + 1e-9)))

        # Bollinger Bands
        bb_mid   = close.rolling(20).mean()
        bb_std   = close.rolling(20).std()
        bb_upper = bb_mid + 2 * bb_std
        bb_lower = bb_mid - 2 * bb_std
        bb_width = (bb_upper - bb_lower) / (bb_mid + 1e-9)

        # MACD
        ema12       = close.ewm(span=12, adjust=False).mean()
        ema26       = close.ewm(span=26, adjust=False).mean()
        macd_line   = ema12 - ema26
        macd_signal = macd_line.ewm(span=9, adjust=False).mean()
        macd_hist   = macd_line - macd_signal

        # Volatility
        daily_return = close.pct_change()
        vol_7d       = daily_return.rolling(7).std()

        # Combine into DataFrame
        result = pd.DataFrame({
            'price'     : close,
            'rsi'       : rsi,
            'bb_width'  : bb_width * 100,     # percentage
            'macd_hist' : macd_hist,
            'volatility': vol_7d * 100,       # percentage
        }).dropna()

        recent = result.tail(30)

        history_list = []
        for index, row in recent.iterrows():
            history_list.append({
                "date"      : index.strftime('%Y-%m-%d'),
                "price"     : round(float(row['price']), 2),
                "rsi"       : round(float(row['rsi']), 2),
                "bb_width"  : round(float(row['bb_width']), 2),
                "macd_hist" : round(float(row['macd_hist']), 2),
                "volatility": round(float(row['volatility']), 2),
            })

        return {"history": history_list}

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
