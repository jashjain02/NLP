import pandas as pd
import numpy as np

from features.features import aggregate_daily_sentiment, merge_with_stock, create_lag_and_roll_features


def test_aggregate_daily_sentiment_counts_and_mean():
    news = pd.DataFrame({
        "trading_date": ["2024-06-03","2024-06-03","2024-06-04"],
        "sent_score": [0.2, -0.1, 0.0],
        "label": ["positive","negative","neutral"],
    })
    agg = aggregate_daily_sentiment(news)
    row = agg[agg["trading_date"] == pd.to_datetime("2024-06-03").date()].iloc[0]
    assert row["daily_count"] == 2
    assert row["daily_pos_count"] == 1
    assert row["daily_neg_count"] == 1
    assert abs(row["daily_mean_sentiment"] - 0.05) < 1e-9


def test_merge_and_targets_no_leak():
    stock = pd.DataFrame({
        "date": pd.date_range("2024-06-03", periods=4, freq="B").date,
        "return": [0.01, -0.02, 0.03, -0.01],
    })
    agg = pd.DataFrame({
        "trading_date": ["2024-06-03","2024-06-04"],
        "daily_mean_sentiment": [0.1, -0.1],
        "daily_count": [5, 2],
        "daily_pos_count": [3, 0],
        "daily_neg_count": [1, 1],
        "daily_neutral_count": [1, 1],
    })
    merged = merge_with_stock(stock, agg)
    assert "target_return_1d" in merged.columns and "target_dir_1d" in merged.columns
    assert merged.loc[0, "target_return_1d"] == stock.loc[1, "return"]


def test_lag_and_roll_features():
    df = pd.DataFrame({
        "date": pd.date_range("2024-06-03", periods=5, freq="B"),
        "daily_mean_sentiment": [0.0, 0.1, -0.2, 0.3, 0.0],
        "daily_count": [1,2,3,1,2],
        "daily_pos_count": [1,1,0,1,0],
        "daily_neg_count": [0,1,2,0,1],
        "daily_neutral_count": [0,0,1,0,1],
    })
    out = create_lag_and_roll_features(df, lags=[1,2], rolls=[3])
    # Lags shifted
    assert np.isnan(out.loc[0, "sent_lag_1"]) and np.isnan(out.loc[1, "sent_lag_2"]) 
    assert out.loc[2, "sent_lag_1"] == df.loc[1, "daily_mean_sentiment"]
    # Rolling windows exist
    assert "sent_roll_3" in out.columns and "count_roll_3" in out.columns
