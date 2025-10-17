#!/usr/bin/env python3
"""
NLP Finance Pipeline Runner
==========================

This script runs the complete NLP Finance pipeline locally.
It can be used for testing or as part of the GitHub Actions workflow.

Usage:
    python run_pipeline.py --ticker AAPL --lookback-days 120
    python run_pipeline.py --config pipeline-config.yaml
"""

import argparse
import sys
import os
from pathlib import Path
import yaml
import pandas as pd
from datetime import datetime, timedelta

# Add project root to path
sys.path.append(str(Path(__file__).parent))

def load_config(config_path=None):
    """Load configuration from file or use defaults."""
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    else:
        # Default configuration
        return {
            'defaults': {
                'ticker': 'AAPL',
                'lookback_days': 120,
                'max_articles': 500
            }
        }

def run_data_collection(ticker, lookback_days, max_articles):
    """Run data collection for stock and news."""
    print(f"📊 Collecting data for {ticker}...")
    
    try:
        # Import data collection modules
        from data_collection.stock import fetch_stock_data
        from data_collection.news import collect_news_multi_source
        
        # Create directories
        os.makedirs('data/raw', exist_ok=True)
        os.makedirs('data/processed', exist_ok=True)
        os.makedirs('models', exist_ok=True)
        os.makedirs('reports/figs', exist_ok=True)
        
        # Collect stock data
        print("  📈 Fetching stock data...")
        stock_df = fetch_stock_data(ticker, lookback_days)
        if stock_df is not None and not stock_df.empty:
            stock_df.to_csv(f'data/raw/{ticker}_stock_data.csv')
            print(f"  ✅ Stock data saved: {len(stock_df)} records")
        else:
            print("  ❌ No stock data collected")
            return False
            
        # Collect news data
        print("  📰 Fetching news data...")
        try:
            news_df = collect_news_multi_source(
                ticker=ticker,
                lookback_days=lookback_days,
                max_articles=max_articles
            )
            if news_df is not None and not news_df.empty:
                news_df.to_csv(f'data/raw/{ticker}_news_data.csv', index=False)
                print(f"  ✅ News data saved: {len(news_df)} articles")
            else:
                print("  ⚠️ No news data collected")
        except Exception as e:
            print(f"  ⚠️ News collection failed: {e}")
            
        return True
        
    except Exception as e:
        print(f"❌ Data collection failed: {e}")
        return False

def run_preprocessing(ticker):
    """Run data preprocessing."""
    print(f"🔧 Preprocessing data for {ticker}...")
    
    try:
        from preprocessing.preprocess import preprocess_data
        
        # Load stock data
        stock_df = pd.read_csv(f'data/raw/{ticker}_stock_data.csv', index_col=0, parse_dates=True)
        
        # Load news data if available
        news_df = None
        news_path = f'data/raw/{ticker}_news_data.csv'
        if os.path.exists(news_path):
            news_df = pd.read_csv(news_path)
            print(f"  📰 Loaded {len(news_df)} news articles")
        
        # Preprocess data
        processed_df = preprocess_data(stock_df, news_df, ticker)
        
        if processed_df is not None and not processed_df.empty:
            processed_df.to_csv(f'data/processed/{ticker}_processed.csv')
            print(f"  ✅ Preprocessed data saved: {len(processed_df)} records")
            return True
        else:
            print("  ❌ Preprocessing failed - no data produced")
            return False
            
    except Exception as e:
        print(f"❌ Preprocessing failed: {e}")
        return False

def run_sentiment_analysis(ticker):
    """Run sentiment analysis."""
    print(f"🧠 Running sentiment analysis for {ticker}...")
    
    try:
        from nlp.sentiment import analyze_sentiment_batch
        
        # Load processed data
        df = pd.read_csv(f'data/processed/{ticker}_processed.csv')
        
        if 'text' in df.columns and not df['text'].isna().all():
            # Run sentiment analysis
            sentiment_df = analyze_sentiment_batch(df)
            
            if sentiment_df is not None:
                sentiment_df.to_csv(f'data/processed/{ticker}_with_sentiment.csv', index=False)
                print(f"  ✅ Sentiment analysis completed: {len(sentiment_df)} records")
                return True
            else:
                print("  ❌ Sentiment analysis failed")
                return False
        else:
            print("  ⚠️ No text data found for sentiment analysis")
            return True  # Continue without sentiment
            
    except Exception as e:
        print(f"❌ Sentiment analysis failed: {e}")
        return False

def run_feature_engineering(ticker):
    """Run feature engineering."""
    print(f"⚙️ Creating features for {ticker}...")
    
    try:
        from features.features import create_features
        
        # Load data with sentiment
        sentiment_file = f'data/processed/{ticker}_with_sentiment.csv'
        if os.path.exists(sentiment_file):
            df = pd.read_csv(sentiment_file)
        else:
            df = pd.read_csv(f'data/processed/{ticker}_processed.csv')
        
        # Create features
        features_df = create_features(df, ticker)
        
        if features_df is not None and not features_df.empty:
            features_df.to_csv(f'data/processed/{ticker}_features.csv', index=False)
            print(f"  ✅ Features created: {len(features_df)} records, {len(features_df.columns)} features")
            return True
        else:
            print("  ❌ Feature engineering failed")
            return False
            
    except Exception as e:
        print(f"❌ Feature engineering failed: {e}")
        return False

def run_model_training(ticker):
    """Run model training."""
    print(f"🤖 Training models for {ticker}...")
    
    try:
        from modeling.modeling import train_classifiers, train_regressors
        import json
        
        # Load features
        df = pd.read_csv(f'data/processed/{ticker}_features.csv')
        
        # Prepare features and targets
        feature_cols = [col for col in df.columns if col not in ['date', 'target_direction', 'target_return']]
        X = df[feature_cols].fillna(0)
        
        # Classification target
        y_clf = df['target_direction'].fillna(0)
        
        # Regression target  
        y_reg = df['target_return'].fillna(0)
        
        print(f"  📊 Training data: {X.shape[0]} samples, {X.shape[1]} features")
        
        # Train classifiers
        print("  🎯 Training classifiers...")
        clf_results = train_classifiers(X, y_clf, ticker)
        
        # Train regressors
        print("  📈 Training regressors...")
        reg_results = train_regressors(X, y_reg, ticker)
        
        # Save results
        results = {
            'ticker': ticker,
            'classifiers': clf_results,
            'regressors': reg_results,
            'data_shape': X.shape,
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        with open(f'reports/metrics_{ticker}.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
            
        print("  ✅ Model training completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Model training failed: {e}")
        return False

def run_backtesting(ticker):
    """Run backtesting."""
    print(f"📊 Running backtest for {ticker}...")
    
    try:
        from backtest.backtest import run_backtest
        import json
        
        # Load features
        df = pd.read_csv(f'data/processed/{ticker}_features.csv')
        
        # Run backtest
        backtest_results = run_backtest(df, ticker)
        
        if backtest_results:
            # Save backtest results
            with open(f'reports/backtest_{ticker}.json', 'w') as f:
                json.dump(backtest_results, f, indent=2, default=str)
                
            print(f"  ✅ Backtest completed: Sharpe={backtest_results.get('sharpe_ratio', 'N/A')}")
            return True
        else:
            print("  ❌ Backtest failed")
            return False
            
    except Exception as e:
        print(f"❌ Backtest failed: {e}")
        return False

def run_evaluation(ticker):
    """Run evaluation and generate reports."""
    print(f"📈 Generating reports for {ticker}...")
    
    try:
        from evaluation.evaluate import generate_diagnostic_plots
        
        # Load data
        df = pd.read_csv(f'data/processed/{ticker}_features.csv')
        
        # Generate plots
        generate_diagnostic_plots(df, ticker)
        
        print("  ✅ Reports generated successfully")
        return True
        
    except Exception as e:
        print(f"❌ Report generation failed: {e}")
        return False

def main():
    """Main pipeline runner."""
    parser = argparse.ArgumentParser(description='Run NLP Finance Pipeline')
    parser.add_argument('--ticker', default='AAPL', help='Stock ticker to analyze')
    parser.add_argument('--lookback-days', type=int, default=120, help='Number of lookback days')
    parser.add_argument('--max-articles', type=int, default=500, help='Maximum articles to collect')
    parser.add_argument('--config', help='Configuration file path')
    parser.add_argument('--skip-data', action='store_true', help='Skip data collection')
    parser.add_argument('--skip-models', action='store_true', help='Skip model training')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override with command line arguments
    ticker = args.ticker
    lookback_days = args.lookback_days
    max_articles = args.max_articles
    
    print(f"🚀 Starting NLP Finance Pipeline")
    print(f"📊 Ticker: {ticker}")
    print(f"📅 Lookback: {lookback_days} days")
    print(f"📰 Max articles: {max_articles}")
    print("=" * 50)
    
    # Pipeline steps
    steps = [
        ("Data Collection", lambda: run_data_collection(ticker, lookback_days, max_articles) if not args.skip_data else True),
        ("Preprocessing", lambda: run_preprocessing(ticker)),
        ("Sentiment Analysis", lambda: run_sentiment_analysis(ticker)),
        ("Feature Engineering", lambda: run_feature_engineering(ticker)),
        ("Model Training", lambda: run_model_training(ticker) if not args.skip_models else True),
        ("Backtesting", lambda: run_backtesting(ticker) if not args.skip_models else True),
        ("Evaluation", lambda: run_evaluation(ticker) if not args.skip_models else True),
    ]
    
    # Run pipeline steps
    success = True
    for step_name, step_func in steps:
        print(f"\n🔄 {step_name}...")
        try:
            if not step_func():
                print(f"❌ {step_name} failed")
                success = False
                break
        except Exception as e:
            print(f"❌ {step_name} failed with error: {e}")
            success = False
            break
    
    # Final status
    print("\n" + "=" * 50)
    if success:
        print("✅ Pipeline completed successfully!")
        print(f"📁 Check the following directories for results:")
        print(f"   - data/processed/{ticker}_features.csv")
        print(f"   - models/")
        print(f"   - reports/")
    else:
        print("❌ Pipeline failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
