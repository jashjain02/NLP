import pandas as pd
import pytest

from preprocessing.preprocess import normalize_timestamps, assign_trading_date, align_news_to_trading_day


def test_normalize_timestamps_makes_utc():
    df = pd.DataFrame({
        "published": [
            "2024-01-01T10:00:00Z",
            "2024-01-01 05:00:00-05:00",
            "2024-01-01 10:00:00",  # naive
        ]
    })
    out = normalize_timestamps(df)
    assert str(out["published"].dtype).startswith("datetime64[ns, UTC]")


def test_assign_trading_date_before_open_weekday():
    # 7:00am NY time on a weekday -> same day trading date
    ts = pd.Timestamp("2024-06-03 07:00:00", tz="America/New_York")
    d = assign_trading_date(ts)
    assert d == pd.Timestamp("2024-06-03").date()


def test_assign_trading_date_during_session():
    ts = pd.Timestamp("2024-06-03 10:00:00", tz="America/New_York")
    d = assign_trading_date(ts)
    assert d == pd.Timestamp("2024-06-03").date()


def test_assign_trading_date_after_close_weekday():
    ts = pd.Timestamp("2024-06-03 17:30:00", tz="America/New_York")
    d = assign_trading_date(ts)
    # Next business day
    assert d == pd.Timestamp("2024-06-04").date()


def test_assign_trading_date_weekend():
    # Saturday -> next business day Monday
    ts = pd.Timestamp("2024-06-01 12:00:00", tz="America/New_York")
    d = assign_trading_date(ts)
    assert d == pd.Timestamp("2024-06-03").date()


def test_align_news_to_trading_day():
    df = pd.DataFrame({
        "title": ["a","b","c","d"],
        "published": [
            "2024-06-03T07:00:00-04:00",  # before open Mon
            "2024-06-03T10:00:00-04:00",  # during
            "2024-06-03T17:30:00-04:00",  # after close
            "2024-06-01T12:00:00-04:00",  # Saturday
        ],
    })
    out = align_news_to_trading_day(df)
    assert list(out["trading_date"]) == [
        pd.Timestamp("2024-06-03").date(),
        pd.Timestamp("2024-06-03").date(),
        pd.Timestamp("2024-06-04").date(),
        pd.Timestamp("2024-06-03").date(),
    ]
