from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

REPORTS_DIR = Path(__file__).resolve().parent.parent / 'reports'
FIGS_DIR = REPORTS_DIR / 'figs'
FIGS_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class BacktestResult:
    equity_curve_path: Path
    metrics: Dict[str, float]


def _compute_drawdown(equity: pd.Series) -> float:
    cum_max = equity.cummax()
    dd = (equity / cum_max) - 1.0
    return float(dd.min())


def run_backtest(
    df: pd.DataFrame,
    pred_col: str = 'prob_up',
    threshold: float = 0.55,
    tc: float = 0.0005,  # transaction cost per trade (round-trip), approx
    position_size: float = 1.0,
) -> Tuple[pd.DataFrame, Dict[str, float], Path]:
    """Run a simple daily backtest.

    - Decision at end of day t uses prob>threshold to go long at t+1 open and exit next open
    - Returns are approximated by daily close-to-close returns in df['return']
    - Transaction costs applied on position changes; slippage absorbed in tc
    """
    if pred_col not in df.columns:
        raise KeyError(f"Missing prediction column: {pred_col}")
    if 'return' not in df.columns:
        raise KeyError("Dataframe must include 'return' daily returns")

    data = df.copy().sort_values('date').reset_index(drop=True)

    prob = data[pred_col].astype(float).fillna(0.0)
    signal = (prob > threshold).astype(int) * position_size

    # Position decided at t based on signal at t, applied to return at t+1
    position = signal.shift(1).fillna(0.0)

    # Strategy gross return
    strat_ret = position * data['return']

    # Transaction costs when position changes (round-trip proxy)
    pos_change = position.diff().abs().fillna(position.abs())
    costs = pos_change * tc

    net_ret = strat_ret - costs

    equity = (1.0 + net_ret).cumprod()

    # Metrics
    ann_factor = 252
    mu = net_ret.mean() * ann_factor
    sigma = net_ret.std(ddof=1) * np.sqrt(ann_factor)
    sharpe = float(mu / sigma) if sigma > 0 else float('nan')
    mdd = _compute_drawdown(equity)
    total_return = float(equity.iloc[-1] - 1.0)

    # Plot equity vs buy & hold
    bh = (1.0 + data['return'].fillna(0.0)).cumprod()
    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(data['date'], equity, label='Strategy')
    ax.plot(data['date'], bh, label='Buy&Hold', alpha=0.7)
    ax.set_title('Equity Curve')
    ax.legend()
    fig.tight_layout()
    out = FIGS_DIR / 'backtest_equity.png'
    fig.savefig(out, dpi=150)
    plt.close(fig)

    metrics = {
        'total_return': total_return,
        'annualized_sharpe': sharpe,
        'max_drawdown': mdd,
        'trades': float((pos_change > 0).sum()),
    }

    out_df = data.copy()
    out_df['position'] = position
    out_df['strategy_return'] = net_ret
    out_df['equity'] = equity

    return out_df, metrics, out


if __name__ == '__main__':
    # Example usage with synthetic data
    n = 100
    dates = pd.date_range('2024-01-01', periods=n, freq='B')
    rng = np.random.default_rng(0)
    ret = rng.normal(0, 0.01, size=n)
    prob = rng.uniform(0.4, 0.6, size=n)
    df = pd.DataFrame({'date': dates, 'return': ret, 'prob_up': prob})

    out_df, metrics, path = run_backtest(df)
    print('Metrics:', metrics)
    print('Equity plot saved to:', path)
