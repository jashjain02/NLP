import os
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import pytest

import builtins
from unittest import mock

from data_collection.stock import fetch_stock_data, _cache_path_for, CACHE_DIR


@pytest.fixture(autouse=True)
def _ensure_cache_dir(tmp_path, monkeypatch):
    # Redirect cache directory to a temp path to avoid polluting repo
    tmp_cache = tmp_path / "data" / "raw"
    tmp_cache.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr("data_collection.stock.CACHE_DIR", tmp_cache, raising=True)
    yield


def _make_fake_df():
    idx = pd.date_range("2020-01-01", periods=5, freq="D", tz="UTC")
    df = pd.DataFrame({
        "Open": [1,2,3,4,5],
        "High": [2,3,4,5,6],
        "Low":  [0,1,2,3,4],
        "Close": [1.5,2.5,3.5,4.5,5.5],
        "Adj Close": [1.4,2.4,3.4,4.4,5.4],
        "Volume": [100,200,300,400,500],
    }, index=idx)
    return df


def _make_multiindex_df():
    idx = pd.date_range("2020-01-01", periods=3, freq="D", tz="UTC")
    cols = pd.MultiIndex.from_product([["Open","High","Low","Close","Adj Close","Volume"],["AAPL"]])
    data = np.array([
        [1,2,0,1.5,1.4,100],
        [2,3,1,2.5,2.4,200],
        [3,4,2,3.5,3.4,300],
    ])
    df = pd.DataFrame(data, index=idx, columns=cols)
    return df


def test_fetch_stock_data_normalizes_and_adds_columns(monkeypatch):
    fake_df = _make_fake_df()

    def fake_download(*args, **kwargs):
        return fake_df.copy()

    with mock.patch("yfinance.download", side_effect=fake_download) as m:
        df = fetch_stock_data("AAPL", "2020-01-01", "2020-12-31")

    assert {"date","open","high","low","close","adj_close","volume","return","log_return"}.issubset(df.columns)
    assert df["date"].dtype == "datetime64[ns]"
    assert len(df) == 5


def test_normalization_handles_multiindex_columns(monkeypatch):
    fake_df = _make_multiindex_df()

    def fake_download(*args, **kwargs):
        return fake_df.copy()

    with mock.patch("yfinance.download", side_effect=fake_download) as m:
        df = fetch_stock_data("AAPL", "2020-01-01", "2020-12-31")

    assert {"date","open","high","low","close","adj_close","volume","return","log_return"}.issubset(df.columns)
    assert len(df) == 3


def test_caching_behavior(monkeypatch, tmp_path):
    fake_df = _make_fake_df()

    def fake_download(*args, **kwargs):
        return fake_df.copy()

    # First call writes cache
    with mock.patch("yfinance.download", side_effect=fake_download) as m:
        df1 = fetch_stock_data("MSFT", "2020-01-01", "2020-12-31")
        assert m.call_count == 1

    # Make cache appear fresh
    cache_path = _cache_path_for("MSFT")
    assert cache_path.exists()

    # Second call should use cache and not call yfinance again
    with mock.patch("yfinance.download", side_effect=fake_download) as m:
        df2 = fetch_stock_data("MSFT", "2020-01-01", "2020-12-31")
        assert m.call_count == 0
        assert len(df2) == len(df1)


def test_invalid_inputs():
    with pytest.raises(ValueError):
        fetch_stock_data("", "2020-01-01", "2020-12-31")
    with pytest.raises(ValueError):
        fetch_stock_data("AAPL", "2020-01-01", "2019-01-01")
    with pytest.raises(ValueError):
        fetch_stock_data("AAPL", "bad", "2020-01-02")
