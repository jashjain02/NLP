from __future__ import annotations

from datetime import date as date_type, datetime, time as time_type
from typing import Optional

import pandas as pd
import numpy as np

try:  # Optional calendar
    import pandas_market_calendars as mcal  # type: ignore
except Exception:  # pragma: no cover
    mcal = None  # type: ignore


def normalize_timestamps(df: pd.DataFrame, timestamp_col: str = "published") -> pd.DataFrame:
    """Ensure a dataframe's timestamp column is timezone-aware in UTC.

    - Accepts strings, naive, or tz-aware datetimes.
    - Returns a copy with the column converted to pandas.Timestamp with UTC tz.
    """
    out = df.copy()
    if timestamp_col not in out.columns:
        raise KeyError(f"Column '{timestamp_col}' not found in dataframe")
    out[timestamp_col] = pd.to_datetime(out[timestamp_col], utc=True, errors="coerce")
    return out


def _parse_hhmm(hhmm: str) -> time_type:
    hh, mm = hhmm.split(":")
    return time_type(int(hh), int(mm))


def _is_session_day(d: pd.Timestamp, market: Optional[str] = "XNYS") -> bool:
    if mcal is None:
        # Fallback: Mon-Fri
        return d.weekday() < 5
    cal = mcal.get_calendar(market or "XNYS")
    # Use timezone-naive dates to avoid tz inference assertion issues
    d_naive = d.normalize().tz_localize(None) if d.tzinfo is not None else d.normalize()
    valid = cal.valid_days(start_date=d_naive, end_date=d_naive)
    return len(valid) > 0


def _next_session_day(d: pd.Timestamp, market: Optional[str] = "XNYS") -> pd.Timestamp:
    if mcal is None:
        return (d + pd.offsets.BDay(1)).normalize()
    cal = mcal.get_calendar(market or "XNYS")
    # Get a small window forward and pick the next valid day strictly after d
    start_naive = d.normalize().tz_localize(None) if d.tzinfo is not None else d.normalize()
    end_naive = (d.normalize() + pd.Timedelta(days=10)).tz_localize(None) if d.tzinfo is not None else d.normalize() + pd.Timedelta(days=10)
    ahead = cal.valid_days(start_date=start_naive, end_date=end_naive)
    for day in ahead:
        if pd.Timestamp(day).normalize() > d.normalize():
            return pd.Timestamp(day).normalize()
    # Fallback to business day if calendar fails
    return (d + pd.offsets.BDay(1)).normalize()


def assign_trading_date(
    ts: pd.Timestamp,
    market_tz: str = "America/New_York",
    market_open: str = "09:30",
    market_close: str = "16:00",
) -> date_type:
    """Assign the trading date for a timestamp according to market hours.

    Rules (local to market_tz):
    - If local time < market_open => assign same calendar date (affects open of D)
    - If market_open <= time <= market_close => assign same trading day D
    - If time > market_close or the day is not a trading session (e.g., weekend/holiday),
      assign the next business trading day.

    Returns a date (no timezone) as pandas.Timestamp.date.
    """
    if not isinstance(ts, pd.Timestamp):
        ts = pd.to_datetime(ts, utc=True)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")

    local_ts = ts.tz_convert(market_tz)
    local_date = local_ts.normalize()

    open_t = _parse_hhmm(market_open)
    close_t = _parse_hhmm(market_close)

    # If not a session day, move to next session day
    if not _is_session_day(local_date):
        next_day = _next_session_day(local_date)
        return pd.Timestamp(next_day).date()

    local_time = local_ts.timetz()
    if local_time < open_t:
        # Before open belongs to today's trading session (same date)
        return pd.Timestamp(local_date).date()
    if local_time <= close_t:
        # During session belongs to today's trading session
        return pd.Timestamp(local_date).date()

    # After close → next session day
    next_day = _next_session_day(local_date)
    return pd.Timestamp(next_day).date()


def align_news_to_trading_day(
    news_df: pd.DataFrame,
    timestamp_col: str = "published",
    market_tz: str = "America/New_York",
    market_open: str = "09:30",
    market_close: str = "16:00",
) -> pd.DataFrame:
    """Add a 'trading_date' column aligning each news item to a trading day.

    - Normalizes timestamps to tz-aware UTC first.
    - Applies assign_trading_date using market hours in market_tz.
    """
    df = normalize_timestamps(news_df, timestamp_col=timestamp_col).copy()
    df["trading_date"] = df[timestamp_col].apply(
        lambda x: assign_trading_date(
            pd.Timestamp(x),
            market_tz=market_tz,
            market_open=market_open,
            market_close=market_close,
        )
    )
    return df


def preprocess_data(stock_df: pd.DataFrame, news_df: Optional[pd.DataFrame] = None, ticker: str = "UNKNOWN") -> pd.DataFrame:
    """
    Preprocess stock and news data for the NLP Finance pipeline.
    
    Args:
        stock_df: Stock price data DataFrame
        news_df: News data DataFrame (optional)
        ticker: Stock ticker symbol
        
    Returns:
        Preprocessed DataFrame with aligned data
    """
    print(f"Preprocessing data for {ticker}...")
    
    # Ensure stock data has proper datetime index
    if not isinstance(stock_df.index, pd.DatetimeIndex):
        if 'date' in stock_df.columns:
            stock_df = stock_df.set_index('date')
        elif 'Date' in stock_df.columns:
            stock_df = stock_df.set_index('Date')
        else:
            # Create a date range if no date column
            stock_df.index = pd.date_range(start='2020-01-01', periods=len(stock_df), freq='D')
    
    # Ensure index is timezone-aware
    if stock_df.index.tz is None:
        stock_df.index = stock_df.index.tz_localize('UTC')
    
    # Create basic features from stock data
    processed_df = stock_df.copy()
    
    # Add basic technical indicators
    if 'Close' in processed_df.columns:
        processed_df['returns'] = processed_df['Close'].pct_change()
        processed_df['log_returns'] = np.log(processed_df['Close'] / processed_df['Close'].shift(1))
        processed_df['volatility'] = processed_df['returns'].rolling(window=20).std()
        processed_df['sma_5'] = processed_df['Close'].rolling(window=5).mean()
        processed_df['sma_20'] = processed_df['Close'].rolling(window=20).mean()
        processed_df['rsi'] = calculate_rsi(processed_df['Close'])
        
        # Create target variables
        processed_df['target_direction'] = (processed_df['returns'] > 0).astype(int)
        processed_df['target_return'] = processed_df['returns']
    
    # Process news data if available
    if news_df is not None and not news_df.empty:
        print(f"Processing {len(news_df)} news articles...")
        
        # Align news to trading days
        if 'published' in news_df.columns:
            news_df = align_news_to_trading_day(news_df, timestamp_col='published')
            
            # Aggregate news by trading date
            news_agg = news_df.groupby('trading_date').agg({
                'title': 'count',
                'text': lambda x: ' '.join(x.dropna().astype(str))
            }).rename(columns={'title': 'news_count', 'text': 'combined_text'})
            
            # Merge with stock data
            processed_df = processed_df.join(news_agg, how='left')
            processed_df['news_count'] = processed_df['news_count'].fillna(0)
            processed_df['combined_text'] = processed_df['combined_text'].fillna('')
        else:
            print("⚠️ No 'published' column found in news data")
    else:
        print("⚠️ No news data provided")
        processed_df['news_count'] = 0
        processed_df['combined_text'] = ''
    
    # Fill missing values
    processed_df = processed_df.fillna(method='ffill').fillna(0)
    
    # Add date column for easier handling
    processed_df['date'] = processed_df.index.date
    
    print(f"✅ Preprocessing completed: {len(processed_df)} records")
    return processed_df


def calculate_rsi(prices: pd.Series, window: int = 14) -> pd.Series:
    """Calculate RSI (Relative Strength Index)."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi
