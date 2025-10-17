import numpy as np
import pandas as pd

from backtest.backtest import run_backtest


def test_backtest_no_trades():
    # Probabilities never exceed threshold -> no trades
    n = 20
    dates = pd.date_range('2024-01-01', periods=n, freq='B')
    ret = np.random.normal(0, 0.01, size=n)
    prob = np.full(n, 0.5)
    df = pd.DataFrame({'date': dates, 'return': ret, 'prob_up': prob})

    out_df, metrics, _ = run_backtest(df, pred_col='prob_up', threshold=0.6, tc=0.0)
    assert metrics['trades'] == 0.0
    # Strategy should be flat (equity ~ 1)
    assert abs(out_df['equity'].iloc[-1] - 1.0) < 1e-9


def test_backtest_simple_profit_case():
    # Always trade when return positive next day: simulate high prob after good returns
    n = 10
    dates = pd.date_range('2024-01-01', periods=n, freq='B')
    ret = np.array([0.01]*n)
    # Set high prob so we are always long (after shift, first day no position)
    prob = np.ones(n)
    df = pd.DataFrame({'date': dates, 'return': ret, 'prob_up': prob})

    out_df, metrics, _ = run_backtest(df, pred_col='prob_up', threshold=0.55, tc=0.0)
    # Equity should grow roughly (1+0.01)^(n-1)
    expected = (1+0.01)**(n-1)
    assert abs(out_df['equity'].iloc[-1] - expected) < 1e-6
    assert metrics['total_return'] > 0
