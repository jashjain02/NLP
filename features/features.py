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
    """Create lag and rolling features for daily_mean_sentiment and counts.

    - Adds sent_lag_k and sent_roll_w features
    - Also adds rolling sums for counts
    """
    if lags is None:
        lags = [1, 2, 3]
    if rolls is None:
        rolls = [3, 7]

    out = df.copy()
    out = out.sort_values("date").reset_index(drop=True)

    # Lags for sentiment
    for k in lags:
        out[f"sent_lag_{k}"] = out["daily_mean_sentiment"].shift(k)

    # Rolling means for sentiment and rolling sums for counts
    for w in rolls:
        out[f"sent_roll_{w}"] = out["daily_mean_sentiment"].rolling(window=w, min_periods=1).mean()
        out[f"count_roll_{w}"] = out["daily_count"].rolling(window=w, min_periods=1).sum()
        out[f"pos_count_roll_{w}"] = out["daily_pos_count"].rolling(window=w, min_periods=1).sum()
        out[f"neg_count_roll_{w}"] = out["daily_neg_count"].rolling(window=w, min_periods=1).sum()
        out[f"neu_count_roll_{w}"] = out["daily_neutral_count"].rolling(window=w, min_periods=1).sum()

    # Sanity: lag columns must be NaN in first k rows; target is shifted forward
    for k in lags:
        assert out[f"sent_lag_{k}"].iloc[:k].isna().all()

    return out


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical indicators to improve model performance."""
    df = df.copy()
    
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
