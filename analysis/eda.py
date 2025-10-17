#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.tsa.stattools import ccf
from statsmodels.tsa.stattools import grangercausalitytests

PROC_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"
FIGS_DIR = REPORTS_DIR / "figs"
FIGS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def _load_features(ticker: str) -> Tuple[pd.DataFrame, str]:
    path = PROC_DIR / f"features_{ticker.upper()}.parquet"
    if path.exists():
        return pd.read_parquet(path), ticker.upper()
    # Fallback: generate synthetic data
    n = 120
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    rng = np.random.default_rng(42)
    sent = rng.normal(0, 0.1, size=n).cumsum() / 5
    returns = rng.normal(0, 0.01, size=n) + 0.2 * np.concatenate([[0], np.diff(sent)])
    prices = 100 * (1 + pd.Series(returns, index=dates)).cumprod()
    df = pd.DataFrame({
        "date": dates,
        "adj_close": prices,
        "daily_mean_sentiment": pd.Series(sent, index=dates).rolling(3, min_periods=1).mean(),
        "return": returns,
        "target_return_1d": pd.Series(returns, index=dates).shift(-1),
    })
    return df, "SAMPLE"


def plot_dual_axis(df: pd.DataFrame, ticker: str) -> Path:
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()
    ax1.plot(df["date"], df["adj_close"], color="tab:blue", label="Adj Close")
    ax2.plot(df["date"], df["daily_mean_sentiment"], color="tab:orange", alpha=0.7, label="Daily Sentiment")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Adj Close", color="tab:blue")
    ax2.set_ylabel("Daily Sentiment", color="tab:orange")
    ax1.set_title(f"{ticker}: Adjusted Close vs Daily Sentiment")
    fig.tight_layout()
    out = FIGS_DIR / f"{ticker}_dual_axis.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_scatter_with_fit(df: pd.DataFrame, ticker: str) -> Path:
    x = df["daily_mean_sentiment"].values
    y = df["target_return_1d"].values
    mask = (~np.isnan(x)) & (~np.isnan(y))
    x, y = x[mask], y[mask]
    r, p = stats.pearsonr(x, y) if len(x) > 2 else (np.nan, np.nan)

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.regplot(x=x, y=y, ax=ax, scatter_kws={"alpha":0.6})
    ax.set_xlabel("Daily Mean Sentiment")
    ax.set_ylabel("Next-day Return")
    ax.set_title(f"{ticker}: Sentiment vs Next-day Return\nPearson r={r:.3f}, p={p:.3g}")
    fig.tight_layout()
    out = FIGS_DIR / f"{ticker}_scatter_fit.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_corr_heatmap(df: pd.DataFrame, ticker: str) -> Path:
    cols = [
        "adj_close",
        "daily_mean_sentiment",
        "daily_count" if "daily_count" in df.columns else None,
        "return",
        "target_return_1d",
    ]
    cols = [c for c in cols if c is not None]
    corr = df[cols].corr()
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title(f"{ticker}: Correlation Heatmap")
    fig.tight_layout()
    out = FIGS_DIR / f"{ticker}_corr_heatmap.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def compute_correlations(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["daily_mean_sentiment", "return", "target_return_1d"]
    res = []
    x = df["daily_mean_sentiment"]
    for col in ["return", "target_return_1d"]:
        y = df[col]
        mask = x.notna() & y.notna()
        if mask.sum() > 2:
            r, p = stats.pearsonr(x[mask], y[mask])
            rho, p_s = stats.spearmanr(x[mask], y[mask])
        else:
            r = p = rho = p_s = np.nan
        res.append({"metric": f"sentiment vs {col}", "pearson_r": r, "pearson_p": p, "spearman_rho": rho, "spearman_p": p_s})
    return pd.DataFrame(res)


def compute_and_plot_ccf(df: pd.DataFrame, ticker: str, max_lag: int = 10) -> Path:
    x = df["daily_mean_sentiment"].fillna(0.0).values
    y = df["return"].fillna(0.0).values
    # ccf gives correlation at lag k where x leads y by k
    vals = ccf(x - x.mean(), y - y.mean(), adjusted=False)
    lags = np.arange(len(vals))[: 2 * max_lag + 1]
    vals = vals[: 2 * max_lag + 1]
    lags = lags - max_lag

    best_idx = int(np.nanargmax(np.abs(vals)))
    best_lag = int(lags[best_idx])

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(lags, vals, color="tab:purple")
    ax.axvline(0, color="black", linewidth=1)
    ax.set_xlabel("Lag (sentiment leads +)")
    ax.set_ylabel("Cross-correlation")
    ax.set_title(f"{ticker}: CCF (max at lag {best_lag})")
    fig.tight_layout()
    out = FIGS_DIR / f"{ticker}_ccf.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def run_granger(df: pd.DataFrame, ticker: str, maxlag: int = 5) -> Path:
    # Use next-day target for stationarity proxy; still this is illustrative
    data = df[["return", "daily_mean_sentiment"]].dropna()
    res = grangercausalitytests(data[["return", "daily_mean_sentiment"]], maxlag=maxlag, verbose=False)
    lines = ["Granger Causality Tests: daily_mean_sentiment -> return"]
    for lag in range(1, maxlag + 1):
        test = res[lag][0]
        pvals = {k: v[1] for k, v in test.items()}  # kpss, ssr_ftest, etc.
        lines.append(f"lag={lag}: p-values {pvals}")
    out_path = REPORTS_DIR / f"granger_{ticker}.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
        f.write("\nInterpretation: If several p-values < 0.05 for ssr_ftest/ssr_chi2test, sentiment Granger-causes returns at that lag.\n")
    return out_path


def main():
    ap = argparse.ArgumentParser(description="EDA for sentiment and returns")
    ap.add_argument("--ticker", required=False, default="AAPL")
    args = ap.parse_args()

    df, used_ticker = _load_features(args.ticker)
    # Heuristics: ensure required columns exist
    if "adj_close" not in df.columns:
        # derive price from close if available
        if "close" in df.columns:
            df["adj_close"] = df["close"]
        else:
            df["adj_close"] = np.nan
    if "target_return_1d" not in df.columns and "return" in df.columns:
        df = df.sort_values("date").reset_index(drop=True)
        df["target_return_1d"] = df["return"].shift(-1)

    # Plots
    p1 = plot_dual_axis(df, used_ticker)
    p2 = plot_scatter_with_fit(df, used_ticker)
    p3 = plot_corr_heatmap(df, used_ticker)
    p4 = compute_and_plot_ccf(df, used_ticker)
    p5 = run_granger(df, used_ticker)

    # Stats summary CSV
    corr_df = compute_correlations(df)
    corr_out = REPORTS_DIR / f"correlations_{used_ticker}.csv"
    corr_df.to_csv(corr_out, index=False)

    print("Saved:")
    for p in [p1, p2, p3, p4, p5, corr_out]:
        print(p)


if __name__ == "__main__":
    main()
