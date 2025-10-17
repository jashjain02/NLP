from __future__ import annotations

from datetime import date as date_type, datetime, time as time_type
from typing import Optional

import pandas as pd

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
