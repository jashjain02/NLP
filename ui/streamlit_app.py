from __future__ import annotations

import os
from datetime import date, timedelta
import sys
from pathlib import Path
import json

import numpy as np
import pandas as pd
import streamlit as st

# Ensure project root is on sys.path so `data_collection` et al. resolve
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_collection.stock import fetch_stock_data
from data_collection.news import (
    fetch_news_rss_and_scrape,
    fetch_news_newsapi,
    fetch_news_moneycontrol_and_scrape,
    fetch_news_marketaux,
)
from preprocessing.preprocess import align_news_to_trading_day
from nlp.sentiment import load_finbert, score_texts_finbert
from features.features import (
    aggregate_daily_sentiment,
    merge_with_stock,
    create_lag_and_roll_features,
    add_technical_indicators,
)
from modeling.modeling import create_time_series_splits
import joblib


st.set_page_config(page_title="NLP Finance - Streamlit", layout="centered")
st.title("Stock Direction Prediction (Streamlit)")


@st.cache_resource
def get_finbert_pipeline():
    return load_finbert()


@st.cache_data(show_spinner=False)
def get_stock(ticker: str, start: str, end: str) -> pd.DataFrame:
    return fetch_stock_data(ticker, start, end, interval='1d')


@st.cache_data(show_spinner=False)
def resolve_company_terms(ticker: str) -> dict:
    """Resolve helpful query terms for a ticker (company name if available)."""
    try:
        import yfinance as yf
        info = (yf.Ticker(ticker).info) or {}
        long_name = info.get('longName') or info.get('shortName')
    except Exception:
        long_name = None
    base = ticker.split('.')[0] if '.' in ticker else ticker
    terms = {"ticker": ticker, "base": base, "company": long_name}
    return terms


@st.cache_data(show_spinner=False)
def get_news_auto(query: str, start: str, end: str, ticker: str, max_articles: int = 600) -> pd.DataFrame:
    """Fetch news from multiple sources and combine results.
    Order: MoneyControl (India-focused), RSS (global), NewsAPI (if key present).
    Deduplicate by URL and filter by date range.
    """
    frames: list[pd.DataFrame] = []
    try:
        mc = fetch_news_moneycontrol_and_scrape(query, start, end, max_articles=max_articles)
        if mc is not None and len(mc) > 0:
            frames.append(mc)
    except Exception:
        pass
    try:
        rss = fetch_news_rss_and_scrape(query, max_articles=max_articles)
        if rss is not None and len(rss) > 0:
            frames.append(rss)
    except Exception:
        pass
    try:
        api_key = os.getenv('NEWSAPI_KEY')
        if api_key:
            na = fetch_news_newsapi(api_key, query, start, end, page_size=min(100, max_articles))
            if na is not None and len(na) > 0:
                frames.append(na)
    except Exception:
        pass
    # MarketAux (use MARKETAUX_API_KEY or NEWSAPI_KEY fallback)
    try:
        mk_key = os.getenv('MARKETAUX_API_KEY')
        if mk_key:
            base_sym = ticker.split('.')[0]
            sym_list = sorted({ticker.upper(), base_sym.upper()})
            mk = fetch_news_marketaux(mk_key, query, start, end, max_results=min(200, max_articles), symbols=sym_list)
            if mk is not None and len(mk) > 0:
                frames.append(mk)
    except Exception:
        pass

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True, sort=False)
    # Normalize and filter by date (make everything tz-aware UTC)
    if 'published' in df.columns:
        df['published'] = pd.to_datetime(df['published'], errors='coerce', utc=True)
        df = df.dropna(subset=['published'])
        start_ts = pd.to_datetime(start, utc=True)
        end_ts = pd.to_datetime(end, utc=True) + pd.Timedelta(hours=23, minutes=59, seconds=59)
        df = df[(df['published'] >= start_ts) & (df['published'] <= end_ts)]
    # Deduplicate by URL
    if 'url' in df.columns:
        df = df.drop_duplicates(subset=['url'])
    return df.reset_index(drop=True)


@st.cache_resource
def load_classifier() -> object | None:
    models_dir = Path(__file__).resolve().parents[1] / 'models'
    rf_path = models_dir / 'clf_rf.joblib'
    if rf_path.exists():
        return joblib.load(rf_path)
    for p in models_dir.glob('clf_*.joblib'):
        try:
            return joblib.load(p)
        except Exception:
            continue
    return None


with st.sidebar:
    st.header("Inputs")
    ticker = st.text_input("Ticker", value="AAPL", help="Yahoo Finance symbol, e.g., AAPL, TSLA, RELIANCE.NS")
    today = date.today()
    end_default = today - timedelta(days=1)
    lookback = st.number_input("Lookback days", min_value=10, max_value=730, value=120, step=10)
    end_date = st.date_input("End date", value=end_default, max_value=end_default)
    start_date = end_date - timedelta(days=int(lookback))
    # Auto-generate company-focused query
    terms = resolve_company_terms(ticker)
    auto_query = f"{terms['ticker']} OR {terms['base']}" + (f" OR \"{terms['company']}\"" if terms['company'] else "") + f" OR {terms['base']} stock OR {terms['base']} news"
    use_auto = st.checkbox("Use auto company query (recommended)", value=True)
    if use_auto:
        st.text_area("News query", value=auto_query, height=80, disabled=True)
        query = auto_query
    else:
        query = st.text_area("News query", value=auto_query, height=80)
    run_btn = st.button("Run Prediction")


if run_btn:
    with st.spinner("Fetching data and running pipeline..."):
        start_iso = start_date.isoformat()
        end_iso = end_date.isoformat()

        stock_df = get_stock(ticker, start_iso, end_iso)
        news_df = get_news_auto(query, start_iso, end_iso, ticker)

    if news_df is None or len(news_df) == 0:
        st.warning("No news found in the selected window. Proceeding with zeroed sentiment features.")
        daily_agg = pd.DataFrame(columns=['trading_date','daily_mean_sentiment','daily_count','daily_pos_count','daily_neg_count','daily_neutral_count'])
    else:
        news_aligned = align_news_to_trading_day(news_df, timestamp_col='published')
        pipe = get_finbert_pipeline()
        # Robustly select text field: prefer 'text', then 'content', then 'description', else 'title'
        if 'text' in news_aligned.columns:
            text_series = news_aligned['text']
        elif 'content' in news_aligned.columns:
            text_series = news_aligned['content']
        elif 'description' in news_aligned.columns:
            text_series = news_aligned['description']
        elif 'title' in news_aligned.columns:
            text_series = news_aligned['title']
        else:
            text_series = pd.Series([''] * len(news_aligned))
        texts = text_series.fillna('').astype(str).tolist()
        sent_df = score_texts_finbert(texts, pipe, batch_size=16, cache_source=f"{ticker}_st")
        sent_df['trading_date'] = news_aligned['trading_date'].values
        sent_df['sent_score'] = sent_df['score']
        daily_agg = aggregate_daily_sentiment(sent_df)

    merged = merge_with_stock(stock_df, daily_agg)
    # Use the same comprehensive feature engineering as training
    from features.features import create_features
    feats = create_features(merged, ticker)

    clf = load_classifier()
    if clf is None:
        st.error("No trained classifier found in models/. Please run the training pipeline first.")
    else:
        try:
            # Build feature vector for the last available row
            row = feats[feats['date'] == pd.to_datetime(end_date)].copy()
            if row.empty:
                row = feats.tail(1).copy()

            # Use all feature columns except target and date columns
            feature_cols = [
                c for c in feats.columns if c not in [
                    'date', 'target_direction', 'target_return', 'trading_date',
                    'daily_mean_sentiment', 'daily_count', 'daily_pos_count', 
                    'daily_neg_count', 'daily_neutral_count'
                ]
            ]

            # Align with saved feature names if available
            models_dir = Path(__file__).resolve().parents[1] / 'models'
            feature_names_path = models_dir / 'feature_names.json'
            if feature_names_path.exists():
                with open(feature_names_path, 'r') as f:
                    trained_feature_names = json.load(f)
                # Ensure all trained features are present (fill missing with 0), drop extras
                for col in trained_feature_names:
                    if col not in row.columns:
                        row[col] = 0.0
                row = row.reindex(columns=trained_feature_names + [c for c in row.columns if c not in trained_feature_names])
                feature_cols = trained_feature_names

            X = row[feature_cols].fillna(0.0).values

            prob_up = clf.predict_proba(X)[:,1] if hasattr(clf, 'predict_proba') else clf.decision_function(X)
            prob_up = float(np.clip(prob_up[0], 0.0, 1.0))
            signal = "UP" if prob_up > 0.5 else "DOWN"

            st.subheader("Prediction")
            st.metric(label=f"{ticker} — {end_date.isoformat()}", value=f"{prob_up:.3f}", delta=signal)
        except Exception as e:
            st.error(f"Prediction failed: {e}")

        with st.expander("Show recent features (last 5 rows)"):
            st.dataframe(feats.tail(5))

        with st.expander("Show news samples"):
            if news_df is not None and len(news_df) > 0:
                show_cols = [c for c in ['title','published','url','source'] if c in news_df.columns]
                st.dataframe(news_df[show_cols].tail(10))
            else:
                st.write("No news available.")


