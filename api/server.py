from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from features.features import create_lag_and_roll_features, aggregate_daily_sentiment, merge_with_stock
from preprocessing.preprocess import align_news_to_trading_day
from data_collection.stock import fetch_stock_data
from data_collection.news import fetch_news_rss_and_scrape, fetch_news_newsapi
from nlp.sentiment import load_finbert, score_texts_finbert

app = FastAPI(title="NLP Finance API")
TEMPLATES_DIR = Path(__file__).resolve().parent / 'templates'
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

MODELS_DIR = Path(__file__).resolve().parent.parent / 'models'
PROC_DIR = Path(__file__).resolve().parent.parent / 'data' / 'processed'


class PredictRequest(BaseModel):
    ticker: str = Field(..., min_length=1)
    date: str = Field(..., pattern=r"^\d{4}-\d{2}-\d{2}$")
    use_newsapi: Optional[bool] = False


class PredictResponse(BaseModel):
    ticker: str
    date: str
    probability_up: float
    signal: int


def _load_latest_classifier() -> Optional[object]:
    rf_path = MODELS_DIR / 'clf_rf.joblib'
    if rf_path.exists():
        return joblib.load(rf_path)
    # fallback to any clf_*.joblib
    for p in MODELS_DIR.glob('clf_*.joblib'):
        try:
            return joblib.load(p)
        except Exception:
            continue
    return None


@app.post('/predict', response_model=PredictResponse)
def predict(req: PredictRequest):
    try:
        target_date = datetime.strptime(req.date, "%Y-%m-%d").date()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid date format; expected YYYY-MM-DD")

    clf = _load_latest_classifier()
    if clf is None:
        raise HTTPException(status_code=500, detail="No classifier found in models/")

    # Fetch minimal stock window around date for features
    start = (pd.Timestamp(target_date) - pd.Timedelta(days=30)).date().isoformat()
    end = target_date.isoformat()
    stock_df = fetch_stock_data(req.ticker, start, end, interval='1d')

    # Fetch news
    if req.use_newsapi:
        api_key = os.getenv('NEWSAPI_KEY')
        news_df = fetch_news_newsapi(api_key, req.ticker, start, end, page_size=50)
    else:
        news_df = fetch_news_rss_and_scrape(req.ticker, max_articles=150)

    news_aligned = align_news_to_trading_day(news_df, timestamp_col='published')

    pipe = load_finbert()
    texts = (news_aligned.get('text') if 'text' in news_aligned.columns else news_aligned.get('content')).fillna('').tolist()
    sent_df = score_texts_finbert(texts, pipe, batch_size=16, cache_source=f"{req.ticker}_api")
    sent_df['trading_date'] = news_aligned['trading_date'].values
    sent_df['sent_score'] = sent_df['score']

    daily_agg = aggregate_daily_sentiment(sent_df)
    merged = merge_with_stock(stock_df, daily_agg)
    feats = create_lag_and_roll_features(merged)

    # Build feature vector for the target_date row
    row = feats[feats['date'] == pd.to_datetime(target_date)].copy()
    if row.empty:
        # choose the last available row as proxy
        row = feats.tail(1).copy()
    feature_cols = [c for c in feats.columns if c.startswith('sent_') or c.startswith('count_roll_') or c in ['daily_mean_sentiment','daily_count']]
    X = row[feature_cols].fillna(0.0).values

    prob_up = clf.predict_proba(X)[:,1] if hasattr(clf, 'predict_proba') else clf.decision_function(X)
    prob_up = float(np.clip(prob_up[0], 0.0, 1.0))
    signal = 1 if prob_up > 0.5 else 0

    return PredictResponse(ticker=req.ticker.upper(), date=target_date.isoformat(), probability_up=prob_up, signal=signal)


@app.get('/', response_class=HTMLResponse)
def ui_index(request: Request):
    return templates.TemplateResponse('index.html', {"request": request})


@app.post('/predict/ui', response_class=HTMLResponse)
def ui_predict(request: Request, ticker: str = Form(...), date: str = Form(...)):
    # Reuse API logic
    api_req = PredictRequest(ticker=ticker, date=date)
    try:
        resp = predict(api_req)
        return templates.TemplateResponse(
            'result.html',
            {
                "request": request,
                "ticker": resp.ticker,
                "date": resp.date,
                "probability_up": f"{resp.probability_up:.3f}",
                "signal": "UP" if resp.signal == 1 else "DOWN",
            },
        )
    except HTTPException as e:
        return templates.TemplateResponse(
            'result.html',
            {
                "request": request,
                "error": e.detail,
            },
            status_code=e.status_code,
        )
