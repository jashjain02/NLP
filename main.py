#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import os
import random
from pathlib import Path
from datetime import date, timedelta

import numpy as np
import pandas as pd

try:
    import torch  # type: ignore
except Exception:  # pragma: no cover
    torch = None  # type: ignore

try:
    import yaml  # PyYAML
except Exception as e:
    raise SystemExit("PyYAML is required. Please install it (pip install pyyaml)")

from data_collection.stock import fetch_stock_data
from data_collection.news import fetch_news_rss_and_scrape, fetch_news_newsapi
from data_collection.news import fetch_news_gdelt_and_scrape, fetch_news_moneycontrol_and_scrape  # new
from preprocessing.preprocess import align_news_to_trading_day
from nlp.sentiment import load_finbert, score_texts_finbert
from features.features import aggregate_daily_sentiment, merge_with_stock, create_lag_and_roll_features, save_features_for_ticker
from modeling.modeling import train_classifiers, train_regressors
from evaluation.evaluate import eval_classifier, eval_regressor
from backtest.backtest import run_backtest
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, classification_report, confusion_matrix


def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(message)s')


def log_event(event: str, **kwargs):
    payload = {"event": event, **kwargs}
    logging.info(json.dumps(payload))


def set_seeds(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        try:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except Exception:
            pass


def load_config(path: Path) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def main():
    ap = argparse.ArgumentParser(description='End-to-end pipeline')
    ap.add_argument('--config', required=True, help='Path to config.yaml')
    args = ap.parse_args()

    setup_logging()
    cfg = load_config(Path(args.config))
    set_seeds(int(cfg.get('seed', 42)))

    ticker = cfg['ticker']

    # Auto recent window with configurable lookback
    dates_cfg = cfg.get('dates', {})
    if dates_cfg.get('auto_recent_window', True):
        lookback_days = int(dates_cfg.get('lookback_days', 120))
        end_dt = date.today() - timedelta(days=1)
        start_dt = date.today() - timedelta(days=lookback_days)
        start = start_dt.isoformat()
        end = end_dt.isoformat()
        log_event('dates_auto_recent_window', start=start, end=end, lookback_days=lookback_days)
    else:
        start = str(dates_cfg['start'])
        end = str(dates_cfg['end'])

    interval = cfg.get('stock', {}).get('interval', '1d')

    log_event('fetch_stock_start', ticker=ticker)
    stock_df = fetch_stock_data(ticker, start, end, interval)
    log_event('fetch_stock_done', rows=len(stock_df))

    news_mode = cfg.get('news', {}).get('mode', 'rss')  # 'rss' or 'newsapi' or 'gdelt' or 'moneycontrol'
    query = cfg.get('news', {}).get('query', ticker)
    news_df = None
    if news_mode == 'newsapi':
        api_key = cfg.get('news', {}).get('api_key') or os.getenv('NEWSAPI_KEY')
        log_event('fetch_news_newsapi_start', query=query)
        news_df = fetch_news_newsapi(api_key, query, start, end, page_size=cfg.get('news', {}).get('page_size', 100))
    elif news_mode == 'gdelt':
        log_event('fetch_news_gdelt_start', query=query)
        news_df = fetch_news_gdelt_and_scrape(query, start, end, max_articles=cfg.get('news', {}).get('max_articles', 1000))
    elif news_mode == 'moneycontrol':
        log_event('fetch_news_moneycontrol_start', query=query)
        news_df = fetch_news_moneycontrol_and_scrape(query, start, end, max_articles=cfg.get('news', {}).get('max_articles', 1000))
    else:
        log_event('fetch_news_rss_start', query=query)
        news_df = fetch_news_rss_and_scrape(query, max_articles=cfg.get('news', {}).get('max_articles', 300))
        # Fallback to NewsAPI if RSS is empty and key is available
        if (news_df is None or len(news_df) == 0) and os.getenv('NEWSAPI_KEY'):
            log_event('fetch_news_rss_empty_try_newsapi')
            try:
                news_df = fetch_news_newsapi(os.getenv('NEWSAPI_KEY'), query, start, end, page_size=min(100, cfg.get('news', {}).get('page_size', 100)))
            except Exception as e:
                log_event('fetch_news_newsapi_fallback_error', error=str(e))
    log_event('fetch_news_done', rows=0 if news_df is None else len(news_df))

    if news_df is None or len(news_df) == 0:
        log_event('news_empty', note='Proceeding with zeroed sentiment features')
        daily_agg = pd.DataFrame(columns=['trading_date','daily_mean_sentiment','daily_count','daily_pos_count','daily_neg_count','daily_neutral_count'])
    else:
        log_event('preprocess_align_start')
        news_aligned = align_news_to_trading_day(news_df, timestamp_col='published')
        log_event('preprocess_align_done')

        log_event('sentiment_start', model='finbert')
        pipe = load_finbert()
        texts = (news_aligned.get('text') if 'text' in news_aligned.columns else news_aligned.get('content')).fillna('').tolist()
        sent_df = score_texts_finbert(texts, pipe, batch_size=cfg.get('nlp', {}).get('batch_size', 16), cache_source=f"{ticker}_finbert")
        sent_df['trading_date'] = news_aligned['trading_date'].values
        sent_df['sent_score'] = sent_df['score']
        sent_df['label'] = sent_df['label']
        log_event('sentiment_done', rows=len(sent_df))

        log_event('features_aggregate_start')
        daily_agg = aggregate_daily_sentiment(sent_df)

    merged = merge_with_stock(stock_df, daily_agg)
    feats = create_lag_and_roll_features(merged)
    
    # Add technical indicators
    from features.features import add_technical_indicators
    feats = add_technical_indicators(feats)
    
    features_path = save_features_for_ticker(feats, ticker)
    log_event('features_done', path=str(features_path))

    # Train simple models
    log_event('train_start')
    df = feats.sort_values('date').reset_index(drop=True)
    # Include sentiment, technical indicators, and price features
    feature_cols = [c for c in df.columns if 
                   c.startswith('sent_') or c.startswith('count_roll_') or 
                   c in ['daily_mean_sentiment','daily_count'] or
                   c.startswith('ma_') or c.startswith('price_') or c.startswith('volatility_') or 
                   c.startswith('momentum_') or c.startswith('bb_') or c.startswith('rsi_') or
                   c.startswith('macd') or c.startswith('trend_') or c.startswith('volatility_regime') or
                   c in ['high_low_ratio', 'close_open_ratio', 'volume_ratio', 'doji', 'hammer', 'shooting_star']]
    X = df[feature_cols].fillna(0.0).values
    y_cls = df['target_dir_1d'].values
    y_reg = df['target_return_1d'].values

    # Holdout days for evaluation
    holdout_days = int(cfg.get('model', {}).get('holdout_days', 10))
    holdout_days = max(1, min(holdout_days, len(df) - 1)) if len(df) > 1 else 1

    X_train = X[:-holdout_days]
    y_cls_train = y_cls[:-holdout_days]
    y_reg_train = y_reg[:-holdout_days]

    cls_res = train_classifiers(X_train, y_cls_train, n_splits=cfg.get('model', {}).get('cv_splits', 5))
    reg_res = train_regressors(X_train, y_reg_train, n_splits=cfg.get('model', {}).get('cv_splits', 5))
    # Log best CV scores for classifiers/regressors when available
    best_cls_scores = {k: round(float(v.get('best_score', float('nan'))), 4) for k, v in cls_res.items()}
    best_reg_scores = {k: round(float(v.get('best_score', float('nan'))), 4) for k, v in reg_res.items()}
    log_event('train_done', cls=list(cls_res.keys()), reg=list(reg_res.keys()), best_cv_f1=best_cls_scores, best_cv_neg_mse=best_reg_scores)

    # Evaluate the RF classifier if present on last N days
    from glob import glob
    clf_path = next((Path(p) for p in glob(str(Path('models') / 'clf_rf.joblib'))), None)
    if clf_path is not None:
        import joblib
        clf = joblib.load(clf_path)
        X_test = X[-holdout_days:]
        y_test = y_cls[-holdout_days:]
        if hasattr(clf, 'predict_proba'):
            y_prob = clf.predict_proba(X_test)[:, 1]
        else:
            y_prob = clf.decision_function(X_test)
        y_pred = (y_prob >= 0.5).astype(int)
        acc = float((y_pred == y_test).mean()) if len(y_test) > 0 else float('nan')
        # Additional metrics
        try:
            f1 = float(f1_score(y_test, y_pred, zero_division=0))
            prec = float(precision_score(y_test, y_pred, zero_division=0))
            rec = float(recall_score(y_test, y_pred, zero_division=0))
        except Exception:
            f1 = prec = rec = float('nan')
        try:
            auc = float(roc_auc_score(y_test, y_prob))
        except Exception:
            auc = float('nan')
        cm = confusion_matrix(y_test, y_pred).tolist()
        report = classification_report(y_test, y_pred, output_dict=True)
        log_event('eval_holdout', days=holdout_days, accuracy=acc, f1=f1, precision=prec, recall=rec, auc=auc, confusion_matrix=cm)

        # Backtest over entire window
        prob_up_full = clf.predict_proba(X)[:,1] if hasattr(clf, 'predict_proba') else clf.decision_function(X)
        back_df = df[['date','return']].copy()
        back_df['prob_up'] = np.asarray(prob_up_full)
        bt_df, bt_metrics, eq_path = run_backtest(back_df, pred_col='prob_up', threshold=cfg.get('backtest', {}).get('threshold', 0.55), tc=cfg.get('backtest', {}).get('tc', 0.0005))
        log_event('backtest_done', metrics=bt_metrics, equity_path=str(eq_path))

        # Persist consolidated metrics to reports/metrics_{ticker}.json
        reports_dir = Path('reports')
        reports_dir.mkdir(parents=True, exist_ok=True)
        metrics_payload = {
            'ticker': ticker,
            'dates': {'start': start, 'end': end},
            'holdout_days': holdout_days,
            'feature_count': len(feature_cols),
            'cv_best_f1': best_cls_scores,
            'cv_best_neg_mse': best_reg_scores,
            'eval_holdout': {
                'accuracy': acc,
                'f1': f1,
                'precision': prec,
                'recall': rec,
                'auc': auc,
                'confusion_matrix': cm,
                'report': report,
            },
            'backtest': bt_metrics,
        }
        out_json = reports_dir / f"metrics_{ticker}.json"
        with open(out_json, 'w', encoding='utf-8') as f:
            json.dump(metrics_payload, f, indent=2, default=float)
        log_event('metrics_saved', path=str(out_json))

    log_event('pipeline_complete', ticker=ticker)


if __name__ == '__main__':
    main()
