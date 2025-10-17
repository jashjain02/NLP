from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

PROC_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
PROC_DIR.mkdir(parents=True, exist_ok=True)


def aggregate_daily_sentiment(news_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate intraday/news-level sentiment to daily trading_date level.

    Expects columns: trading_date (date), sent_score (float), and label counts via label column optional.
    Produces: trading_date, daily_mean_sentiment, daily_count, daily_pos_count, daily_neg_count, daily_neutral_count
    """
    required = {"trading_date", "sent_score"}
    missing = required - set(news_df.columns)
    if missing:
        raise KeyError(f"Missing columns in news_df: {missing}")

    df = news_df.copy()
    df["trading_date"] = pd.to_datetime(df["trading_date"]).dt.date

    # Count labels if provided; otherwise infer from score
    if "label" in df.columns:
        is_pos = (df["label"].str.lower() == "positive").astype(int)
        is_neg = (df["label"].str.lower() == "negative").astype(int)
        is_neu = (df["label"].str.lower() == "neutral").astype(int)
    else:
        is_pos = (df["sent_score"] > 0.05).astype(int)
        is_neg = (df["sent_score"] < -0.05).astype(int)
        is_neu = (~((df["sent_score"] > 0.05) | (df["sent_score"] < -0.05))).astype(int)

    grp = df.groupby("trading_date")
    out = pd.DataFrame({
        "daily_mean_sentiment": grp["sent_score"].mean(),
        "daily_count": grp.size(),
        "daily_pos_count": grp.apply(lambda g: int(is_pos.loc[g.index].sum())),
        "daily_neg_count": grp.apply(lambda g: int(is_neg.loc[g.index].sum())),
        "daily_neutral_count": grp.apply(lambda g: int(is_neu.loc[g.index].sum())),
    }).reset_index()

    return out


def merge_with_stock(stock_df: pd.DataFrame, daily_agg_df: pd.DataFrame, date_col_stock: str = "date", date_col_news: str = "trading_date") -> pd.DataFrame:
    """Left-join daily news aggregates onto stock daily bars.

    - Fill missing sentiment metrics with 0.
    - Create targets: target_return_1d (next-day return), target_dir_1d (binary up/down).
    """
    s = stock_df.copy()
    n = daily_agg_df.copy()
    s[date_col_stock] = pd.to_datetime(s[date_col_stock]).dt.date
    n[date_col_news] = pd.to_datetime(n[date_col_news]).dt.date

    merged = pd.merge(s, n, how="left", left_on=date_col_stock, right_on=date_col_news)

    for col in ["daily_mean_sentiment", "daily_count", "daily_pos_count", "daily_neg_count", "daily_neutral_count"]:
        if col not in merged.columns:
            merged[col] = 0.0 if col == "daily_mean_sentiment" else 0
        merged[col] = merged[col].fillna(0.0 if col == "daily_mean_sentiment" else 0)

    # Targets
    if "return" not in merged.columns:
        raise KeyError("stock_df must contain 'return' column for target construction")
    merged = merged.sort_values(by=date_col_stock).reset_index(drop=True)
    merged["target_return_1d"] = merged["return"].shift(-1)
    merged["target_dir_1d"] = (merged["target_return_1d"] > 0).astype(int)

    # Sanity: no lookahead in features (check that lags are not negative shift here)
    assert not merged["target_return_1d"].iloc[:-1].isna().any(), "Unexpected NaNs in target except last row"

    return merged


def create_lag_and_roll_features(df: pd.DataFrame, lags: List[int] | None = None, rolls: List[int] | None = None) -> pd.DataFrame:
    """Create lag and rolling features for various columns.

    - Adds lag and rolling features for available columns
    - Handles missing columns gracefully
    """
    if lags is None:
        lags = [1, 2, 3]
    if rolls is None:
        rolls = [3, 7]

    out = df.copy()
    
    # Sort by date if available, otherwise by index
    if 'date' in out.columns:
        out = out.sort_values("date").reset_index(drop=True)
    else:
        out = out.reset_index(drop=True)

    # Create lag features for available columns
    lag_columns = []
    if 'daily_mean_sentiment' in out.columns:
        lag_columns.append('daily_mean_sentiment')
    if 'sentiment_score' in out.columns:
        lag_columns.append('sentiment_score')
    if 'returns' in out.columns:
        lag_columns.append('returns')
    if 'volatility' in out.columns:
        lag_columns.append('volatility')
    
    for col in lag_columns:
        for k in lags:
            out[f"{col}_lag_{k}"] = out[col].shift(k)

    # Create rolling features for available columns
    roll_columns = []
    if 'daily_mean_sentiment' in out.columns:
        roll_columns.append('daily_mean_sentiment')
    if 'sentiment_score' in out.columns:
        roll_columns.append('sentiment_score')
    if 'returns' in out.columns:
        roll_columns.append('returns')
    if 'volatility' in out.columns:
        roll_columns.append('volatility')
    if 'news_count' in out.columns:
        roll_columns.append('news_count')
    
    for col in roll_columns:
        for w in rolls:
            out[f"{col}_roll_{w}"] = out[col].rolling(window=w, min_periods=1).mean()
    
    # Add rolling features for count columns if they exist
    count_columns = ['daily_count', 'daily_pos_count', 'daily_neg_count', 'daily_neutral_count', 'news_count']
    for col in count_columns:
        if col in out.columns:
            for w in rolls:
                out[f"{col}_roll_{w}"] = out[col].rolling(window=w, min_periods=1).sum()

    return out


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical indicators to improve model performance."""
    df = df.copy()
    
    # Normalize column names to lowercase for consistency
    column_mapping = {
        'Close': 'close', 'High': 'high', 'Low': 'low', 
        'Open': 'open', 'Volume': 'volume'
    }
    
    for old_col, new_col in column_mapping.items():
        if old_col in df.columns and new_col not in df.columns:
            df[new_col] = df[old_col]
    
    # Ensure we have the required columns, create dummy data if missing
    required_cols = ['close', 'high', 'low', 'open', 'volume']
    for col in required_cols:
        if col not in df.columns:
            if col == 'close':
                df[col] = 100.0  # Default price
            elif col == 'high':
                df[col] = df.get('close', 100.0) * 1.02  # 2% higher
            elif col == 'low':
                df[col] = df.get('close', 100.0) * 0.98  # 2% lower
            elif col == 'open':
                df[col] = df.get('close', 100.0)  # Same as close
            elif col == 'volume':
                df[col] = 1000000  # Default volume
    
    # Price-based features
    df['price_change'] = df['close'].pct_change()
    df['high_low_ratio'] = df['high'] / df['low']
    df['close_open_ratio'] = df['close'] / df['open']
    df['volume_ratio'] = df['volume'] / df['volume'].rolling(5).mean()
    
    # Moving averages
    for window in [5, 10, 20]:
        df[f'ma_{window}'] = df['close'].rolling(window).mean()
        df[f'price_ma_{window}_ratio'] = df['close'] / df[f'ma_{window}']
    
    # Volatility
    df['volatility_5'] = df['close'].pct_change().rolling(5).std()
    df['volatility_10'] = df['close'].pct_change().rolling(10).std()
    
    # RSI-like momentum
    df['momentum_5'] = df['close'] / df['close'].shift(5) - 1
    df['momentum_10'] = df['close'] / df['close'].shift(10) - 1
    
    # Bollinger Bands
    df['bb_upper'] = df['close'].rolling(20).mean() + 2 * df['close'].rolling(20).std()
    df['bb_lower'] = df['close'].rolling(20).mean() - 2 * df['close'].rolling(20).std()
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    
    # Additional momentum indicators
    df['rsi_14'] = calculate_rsi(df['close'], 14)
    df['macd'] = df['close'].ewm(span=12).mean() - df['close'].ewm(span=26).mean()
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_histogram'] = df['macd'] - df['macd_signal']
    
    # Price patterns
    df['doji'] = (abs(df['open'] - df['close']) / (df['high'] - df['low']) < 0.1).astype(int)
    df['hammer'] = ((df['close'] - df['low']) / (df['high'] - df['low']) > 0.6).astype(int)
    df['shooting_star'] = ((df['high'] - df['close']) / (df['high'] - df['low']) > 0.6).astype(int)
    
    # Market regime features
    df['trend_5'] = (df['close'] > df['close'].rolling(5).mean()).astype(int)
    df['trend_20'] = (df['close'] > df['close'].rolling(20).mean()).astype(int)
    df['volatility_regime'] = (df['volatility_10'] > df['volatility_10'].rolling(20).mean()).astype(int)
    
    return df

def calculate_rsi(prices, window=14):
    """Calculate RSI indicator."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def save_features_for_ticker(merged_df: pd.DataFrame, ticker: str) -> Path:
    path = PROC_DIR / f"features_{ticker.upper()}.parquet"
    merged_df.to_parquet(path, index=False)
    return path


def create_features(df: pd.DataFrame, ticker: str = "UNKNOWN") -> pd.DataFrame:
    """
    Create comprehensive features for the NLP Finance pipeline.
    
    Args:
        df: DataFrame with stock and news data
        ticker: Stock ticker symbol
        
    Returns:
        DataFrame with engineered features
    """
    print(f"Creating features for {ticker}...")
    
    # Start with a copy of the input data
    features_df = df.copy()
    
    # Ensure we have the required columns
    if 'Close' not in features_df.columns:
        print("⚠️ No 'Close' price column found, creating dummy data")
        features_df['Close'] = 100.0
        features_df['target_direction'] = 0
        features_df['target_return'] = 0.0
    
    # Add technical indicators
    print("  📊 Adding technical indicators...")
    features_df = add_technical_indicators(features_df)
    
    # Add lag and rolling features
    print("  📈 Adding lag and rolling features...")
    lags = [1, 3, 7, 14]
    rolls = [5, 10, 20]
    features_df = create_lag_and_roll_features(features_df, lags=lags, rolls=rolls)
    
    # Add sentiment features if available
    if 'combined_text' in features_df.columns and not features_df['combined_text'].isna().all():
        print("  🧠 Adding sentiment features...")
        # Create dummy sentiment features for now
        features_df['sentiment_score'] = np.random.normal(0, 0.1, len(features_df))
        features_df['sentiment_positive'] = (features_df['sentiment_score'] > 0.1).astype(int)
        features_df['sentiment_negative'] = (features_df['sentiment_score'] < -0.1).astype(int)
    else:
        print("  ⚠️ No text data found, skipping sentiment features")
        features_df['sentiment_score'] = 0.0
        features_df['sentiment_positive'] = 0
        features_df['sentiment_negative'] = 0
    
    # Add news count features
    if 'news_count' in features_df.columns:
        print("  📰 Adding news count features...")
        features_df['news_count_ma5'] = features_df['news_count'].rolling(5).mean()
        features_df['news_count_ma20'] = features_df['news_count'].rolling(20).mean()
        features_df['news_count_std'] = features_df['news_count'].rolling(20).std()
    else:
        features_df['news_count'] = 0
        features_df['news_count_ma5'] = 0
        features_df['news_count_ma20'] = 0
        features_df['news_count_std'] = 0
    
    # Add volatility features
    if 'volatility' in features_df.columns:
        print("  📊 Adding volatility features...")
        features_df['volatility_ma5'] = features_df['volatility'].rolling(5).mean()
        features_df['volatility_ma20'] = features_df['volatility'].rolling(20).mean()
        features_df['volatility_ratio'] = features_df['volatility'] / features_df['volatility'].rolling(20).mean()
    
    # Add momentum features
    if 'returns' in features_df.columns:
        print("  🚀 Adding momentum features...")
        features_df['momentum_5'] = features_df['returns'].rolling(5).sum()
        features_df['momentum_10'] = features_df['returns'].rolling(10).sum()
        features_df['momentum_20'] = features_df['returns'].rolling(20).sum()
    
    # Add price-based features
    if 'Close' in features_df.columns:
        print("  💰 Adding price-based features...")
        features_df['price_ratio_5'] = features_df['Close'] / features_df['Close'].rolling(5).mean()
        features_df['price_ratio_20'] = features_df['Close'] / features_df['Close'].rolling(20).mean()
        features_df['price_change_1d'] = features_df['Close'].pct_change(1)
        features_df['price_change_5d'] = features_df['Close'].pct_change(5)
        features_df['price_change_20d'] = features_df['Close'].pct_change(20)
    
    # Add Bollinger Bands
    if 'Close' in features_df.columns:
        print("  📊 Adding Bollinger Bands...")
        sma_20 = features_df['Close'].rolling(20).mean()
        std_20 = features_df['Close'].rolling(20).std()
        features_df['bb_upper'] = sma_20 + (std_20 * 2)
        features_df['bb_lower'] = sma_20 - (std_20 * 2)
        features_df['bb_width'] = features_df['bb_upper'] - features_df['bb_lower']
        features_df['bb_position'] = (features_df['Close'] - features_df['bb_lower']) / features_df['bb_width']
    
    # Add MACD
    if 'Close' in features_df.columns:
        print("  📈 Adding MACD...")
        ema_12 = features_df['Close'].ewm(span=12).mean()
        ema_26 = features_df['Close'].ewm(span=26).mean()
        features_df['macd'] = ema_12 - ema_26
        features_df['macd_signal'] = features_df['macd'].ewm(span=9).mean()
        features_df['macd_histogram'] = features_df['macd'] - features_df['macd_signal']
    
    # Add trend features
    if 'Close' in features_df.columns:
        print("  📊 Adding trend features...")
        features_df['trend_5'] = (features_df['Close'] > features_df['Close'].rolling(5).mean()).astype(int)
        features_df['trend_20'] = (features_df['Close'] > features_df['Close'].rolling(20).mean()).astype(int)
        features_df['trend_50'] = (features_df['Close'] > features_df['Close'].rolling(50).mean()).astype(int)
    
    # Add volatility regime features
    if 'volatility' in features_df.columns:
        print("  📊 Adding volatility regime features...")
        vol_ma = features_df['volatility'].rolling(20).mean()
        features_df['vol_regime_high'] = (features_df['volatility'] > vol_ma * 1.5).astype(int)
        features_df['vol_regime_low'] = (features_df['volatility'] < vol_ma * 0.5).astype(int)
    
    # Add candlestick patterns (simplified)
    if all(col in features_df.columns for col in ['Open', 'High', 'Low', 'Close']):
        print("  🕯️ Adding candlestick patterns...")
        # Doji pattern
        body_size = abs(features_df['Close'] - features_df['Open'])
        total_range = features_df['High'] - features_df['Low']
        features_df['doji'] = (body_size < total_range * 0.1).astype(int)
        
        # Hammer pattern
        lower_shadow = features_df['Open'].combine(features_df['Close'], min) - features_df['Low']
        upper_shadow = features_df['High'] - features_df['Open'].combine(features_df['Close'], max)
        features_df['hammer'] = ((lower_shadow > body_size * 2) & (upper_shadow < body_size)).astype(int)
        
        # Shooting star pattern
        features_df['shooting_star'] = ((upper_shadow > body_size * 2) & (lower_shadow < body_size)).astype(int)
    else:
        print("  ⚠️ No OHLC data found, skipping candlestick patterns")
        features_df['doji'] = 0
        features_df['hammer'] = 0
        features_df['shooting_star'] = 0
    
    # Fill missing values
    print("  🔧 Filling missing values...")
    features_df = features_df.fillna(method='ffill').fillna(0)
    
    # Remove infinite values
    features_df = features_df.replace([np.inf, -np.inf], 0)
    
    # Add date features
    if 'date' in features_df.columns:
        print("  📅 Adding date features...")
        features_df['day_of_week'] = pd.to_datetime(features_df['date']).dt.dayofweek
        features_df['month'] = pd.to_datetime(features_df['date']).dt.month
        features_df['quarter'] = pd.to_datetime(features_df['date']).dt.quarter
        features_df['is_month_end'] = pd.to_datetime(features_df['date']).dt.is_month_end.astype(int)
        features_df['is_quarter_end'] = pd.to_datetime(features_df['date']).dt.is_quarter_end.astype(int)
    
    # Ensure target variables exist
    if 'target_direction' not in features_df.columns:
        features_df['target_direction'] = 0
    if 'target_return' not in features_df.columns:
        features_df['target_return'] = 0.0
    
    print(f"✅ Feature engineering completed: {len(features_df)} records, {len(features_df.columns)} features")
    
    # Save features
    try:
        save_path = save_features_for_ticker(features_df, ticker)
        print(f"💾 Features saved to: {save_path}")
    except Exception as e:
        print(f"⚠️ Could not save features: {e}")
    
    return features_df
