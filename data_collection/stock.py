from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf


CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class RetryConfig:
    max_retries: int = 3
    initial_backoff_seconds: float = 1.0
    backoff_multiplier: float = 2.0


def _validate_inputs(ticker: str, start: str, end: str, interval: str) -> None:
    if not isinstance(ticker, str) or not ticker.strip():
        raise ValueError("ticker must be a non-empty string, e.g., 'AAPL'.")
    if not isinstance(start, str) or not isinstance(end, str):
        raise ValueError("start and end must be ISO date strings, e.g., '2018-01-01'.")
    try:
        start_dt = datetime.fromisoformat(start)
        end_dt = datetime.fromisoformat(end)
    except Exception as exc:
        raise ValueError("start/end must be valid ISO dates, e.g., '2018-01-01'.") from exc
    if end_dt <= start_dt:
        raise ValueError("end date must be after start date.")
    allowed = {"1m","2m","5m","15m","30m","60m","90m","1h","1d","5d","1wk","1mo","3mo"}
    if interval not in allowed:
        raise ValueError(f"interval must be one of {sorted(allowed)}")


def _cache_path_for(ticker: str) -> Path:
    safe_ticker = ticker.upper().replace("/", "-")
    return CACHE_DIR / f"{safe_ticker}.parquet"


def _is_cache_fresh(path: Path, max_age: timedelta = timedelta(days=1)) -> bool:
    if not path.exists():
        return False
    try:
        mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        return datetime.now(timezone.utc) - mtime <= max_age
    except Exception:
        return False


def _load_from_cache(path: Path) -> Optional[pd.DataFrame]:
    try:
        df = pd.read_parquet(path)
        if not isinstance(df, pd.DataFrame) or df.empty:
            return None
        return df
    except Exception:
        return None


def _save_to_cache(path: Path, df: pd.DataFrame) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
    except Exception:
        # Best-effort cache write; ignore errors
        pass


def _flatten_yf_columns(df: pd.DataFrame) -> pd.DataFrame:
    # yfinance may return MultiIndex columns like ('Close','AAPL'). Handle common cases.
    if isinstance(df.columns, pd.MultiIndex):
        if df.columns.nlevels == 2:
            level1 = df.columns.get_level_values(1)
            # If the second level is uniform (single ticker), drop it
            if len(set(level1)) == 1:
                df.columns = df.columns.droplevel(1)
            else:
                # Flatten with underscore join
                df.columns = [f"{a}_{b}" if b else str(a) for a, b in df.columns]
        else:
            # Fallback: join all levels
            df.columns = ["_".join([str(x) for x in tpl if str(x) != ""]) for tpl in df.columns]
    return df


def _normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        raise ValueError("No data returned for given parameters.")

    # Handle potential MultiIndex columns before any renaming
    df = _flatten_yf_columns(df)

    # yfinance may return index as DatetimeIndex; ensure a column 'date' tz-naive UTC
    if isinstance(df.index, pd.DatetimeIndex):
        dt = df.index.tz_convert("UTC") if df.index.tz is not None else df.index.tz_localize("UTC")
        date_col = pd.to_datetime(dt).tz_convert(None)
        df = df.reset_index(drop=False)
        df.insert(0, "date", date_col.tz_localize(None))
    elif "Date" in df.columns:
        date_col = pd.to_datetime(df["Date"], utc=True).dt.tz_convert(None)
        df.insert(0, "date", date_col)
    else:
        # fallback: try to parse any column that looks like date
        for candidate in ["date", "datetime", "timestamp"]:
            if candidate in df.columns:
                date_col = pd.to_datetime(df[candidate], utc=True).dt.tz_convert(None)
                df["date"] = date_col
                break
        else:
            raise ValueError("Could not determine date column from yfinance result.")

    # Standardize column names to lower snake
    rename_map = {
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "adj_close",
        "Adj_Close": "adj_close",
        "Volume": "volume",
        "Stock Splits": "stock_splits",
        "Dividends": "dividends",
    }
    df = df.rename(columns=rename_map)

    # If yfinance flattened produced e.g., 'Close_AAPL', map those too
    for key, val in list(rename_map.items()):
        for suffix in ("_AAPL", "_" + key.replace(" ", "_")):
            if key + suffix in df.columns:
                df = df.rename(columns={key + suffix: val})

    # Ensure expected OHLCV columns exist if possible
    for col in ["open", "high", "low", "close", "volume"]:
        if col not in df.columns:
            df[col] = np.nan

    # Add adj_close if missing
    if "adj_close" not in df.columns:
        df["adj_close"] = df.get("close", pd.Series(np.nan, index=df.index))

    # Compute returns
    df = df.sort_values("date").reset_index(drop=True)
    df["return"] = df["adj_close"].pct_change()
    df["log_return"] = np.log(df["adj_close"]).diff()

    # Reorder columns
    ordered_cols = [
        "date","open","high","low","close","adj_close","volume","return","log_return"
    ]
    remaining = [c for c in df.columns if c not in ordered_cols]
    df = df[ordered_cols + remaining]

    return df


def fetch_stock_data(ticker: str, start: str, end: str, interval: str = "1d", *, retry: RetryConfig | None = None) -> pd.DataFrame:
    """Download OHLCV data via yfinance with caching and retries.

    - Adds columns: date (tz-naive UTC), adj_close, return, log_return.
    - Caches to parquet under data/raw/{ticker}.parquet.
    """
    _validate_inputs(ticker, start, end, interval)

    cache_path = _cache_path_for(ticker)
    if _is_cache_fresh(cache_path):
        cached = _load_from_cache(cache_path)
        if cached is not None:
            return cached

    cfg = retry or RetryConfig()
    attempt = 0
    last_exc: Optional[Exception] = None
    while attempt < cfg.max_retries:
        try:
            df = yf.download(tickers=ticker, start=start, end=end, interval=interval, auto_adjust=False, progress=False, threads=False)
            df = _normalize_dataframe(df)
            _save_to_cache(cache_path, df)
            return df
        except Exception as exc:
            last_exc = exc
            attempt += 1
            if attempt >= cfg.max_retries:
                break
            backoff = cfg.initial_backoff_seconds * (cfg.backoff_multiplier ** (attempt - 1))
            time.sleep(backoff)

    msg = (
        f"Failed to fetch data for {ticker} from {start} to {end} at interval {interval} "
        f"after {cfg.max_retries} retries. Last error: {last_exc}"
    )
    raise RuntimeError(msg) from last_exc


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fetch stock data via yfinance with caching.")
    p.add_argument("--ticker", required=True, help="Ticker symbol, e.g., AAPL")
    p.add_argument("--start", required=True, help="Start date, e.g., 2018-01-01")
    p.add_argument("--end", required=True, help="End date, e.g., 2024-12-31")
    p.add_argument("--interval", default="1d", help="yfinance interval, default 1d")
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)
    try:
        df = fetch_stock_data(args.ticker, args.start, args.end, args.interval)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(df.head().to_string(index=False))
    print(f"\nRows: {len(df)}  Columns: {list(df.columns)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
