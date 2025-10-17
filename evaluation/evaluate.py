#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    mean_squared_error,
    r2_score,
)

REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"
FIGS_DIR = REPORTS_DIR / "figs"
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
FIGS_DIR.mkdir(parents=True, exist_ok=True)


def _load_features(ticker: str) -> pd.DataFrame:
    path = DATA_DIR / f"features_{ticker.upper()}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Features not found: {path}")
    return pd.read_parquet(path)


def _save_json(obj: Dict, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=float)


def eval_classifier(model_path: Path, df: pd.DataFrame, ticker: str) -> Dict:
    model = joblib.load(model_path)
    df = df.sort_values("date").reset_index(drop=True)

    feature_cols = [c for c in df.columns if c.startswith('sent_') or c.startswith('count_roll_') or c in ['daily_mean_sentiment','daily_count']]
    X = df[feature_cols].fillna(0.0).values
    y = df['target_dir_1d'].values

    # Keep last row for holdout demonstration
    X_train, X_test = X[:-1], X[-1:]
    y_train, y_test = y[:-1], y[-1:]

    y_pred_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else model.decision_function(X_test)
    y_pred = (y_pred_prob >= 0.5).astype(int)

    # Confusion matrix and report (on this tiny holdout; for real use, provide a true test set)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    # ROC curve
    try:
        auc = roc_auc_score(y_test, y_pred_prob)
    except Exception:
        auc = float('nan')
    fpr, tpr, th = roc_curve(y_test, y_pred_prob)

    # Plots
    fig, ax = plt.subplots(figsize=(4,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title('Confusion Matrix')
    fig.tight_layout()
    fig.savefig(FIGS_DIR / f"{ticker}_clf_confusion.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5,4))
    ax.plot(fpr, tpr, label=f'AUC={auc:.3f}')
    ax.plot([0,1],[0,1],'k--')
    ax.set_xlabel('FPR')
    ax.set_ylabel('TPR')
    ax.set_title('ROC Curve')
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGS_DIR / f"{ticker}_clf_roc.png", dpi=150)
    plt.close(fig)

    # Feature importances for tree models
    if hasattr(getattr(model, 'named_steps', model), 'clf'):
        clf = getattr(model.named_steps, 'clf', None)
        if clf is None and hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif clf is not None and hasattr(clf, 'feature_importances_'):
            importances = clf.feature_importances_
        else:
            importances = None
        if importances is not None:
            idx = np.argsort(importances)[::-1][:20]
            names = np.array(feature_cols)[idx]
            vals = importances[idx]
            fig, ax = plt.subplots(figsize=(6,4))
            ax.barh(names[::-1], vals[::-1])
            ax.set_title('Top Feature Importances')
            fig.tight_layout()
            fig.savefig(FIGS_DIR / f"{ticker}_clf_importances.png", dpi=150)
            plt.close(fig)

    # SHAP summary (optional)
    try:  # pragma: no cover
        import shap  # type: ignore
        explainer = shap.Explainer(model, X_train)
        shap_values = explainer(X_test)
        shap.plots.beeswarm(shap_values, show=False)
        plt.tight_layout()
        plt.savefig(FIGS_DIR / f"{ticker}_clf_shap.png", dpi=150)
        plt.close()
    except Exception:
        pass

    # Directional accuracy over time on train part
    yhat_train = ( (model.predict_proba(X_train)[:,1] if hasattr(model,'predict_proba') else model.decision_function(X_train)) >= 0.5 ).astype(int)
    acc_series = (yhat_train == y_train).astype(int)
    roll = pd.Series(acc_series).rolling(window=20, min_periods=1).mean()
    fig, ax = plt.subplots(figsize=(8,3))
    ax.plot(roll.values)
    ax.set_title('Rolling Directional Accuracy (train)')
    fig.tight_layout()
    fig.savefig(FIGS_DIR / f"{ticker}_clf_rolling_acc.png", dpi=150)
    plt.close(fig)

    metrics = {
        'auc': float(auc),
        'report': report,
    }
    return metrics


def eval_regressor(model_path: Path, df: pd.DataFrame, ticker: str) -> Dict:
    model = joblib.load(model_path)
    df = df.sort_values("date").reset_index(drop=True)

    feature_cols = [c for c in df.columns if c.startswith('sent_') or c.startswith('count_roll_') or c in ['daily_mean_sentiment','daily_count']]
    X = df[feature_cols].fillna(0.0).values
    y = df['target_return_1d'].values

    X_train, X_test = X[:-1], X[-1:]
    y_train, y_test = y[:-1], y[-1:]

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Residual plot and predicted vs actual
    fig, ax = plt.subplots(1,2, figsize=(8,3))
    resid = y_test - y_pred
    ax[0].scatter(y_pred, resid)
    ax[0].axhline(0, color='k', linestyle='--')
    ax[0].set_title('Residuals vs Predicted')

    ax[1].scatter(y_test, y_pred)
    ax[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
    ax[1].set_title('Predicted vs Actual')

    fig.tight_layout()
    fig.savefig(FIGS_DIR / f"{ticker}_reg_residuals.png", dpi=150)
    plt.close(fig)

    metrics = {
        'mse': float(mse),
        'r2': float(r2),
    }
    return metrics


def main():
    ap = argparse.ArgumentParser(description='Evaluate saved models')
    ap.add_argument('--ticker', required=True)
    ap.add_argument('--model_path', required=True, help='Path to saved joblib model')
    ap.add_argument('--task', choices=['clf','reg'], required=True)
    args = ap.parse_args()

    df = _load_features(args.ticker)

    if args.task == 'clf':
        metrics = eval_classifier(Path(args.model_path), df, args.ticker)
    else:
        metrics = eval_regressor(Path(args.model_path), df, args.ticker)

    # Save metrics
    out_json = REPORTS_DIR / f"metrics_{Path(args.model_path).stem}.json"
    _save_json(metrics, out_json)
    print(f"Saved metrics to {out_json}")


if __name__ == '__main__':
    main()
