#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import argparse
import numpy as np
import pandas as pd

from modeling.modeling import train_classifiers, train_regressors

PROC_DIR = Path(__file__).resolve().parent / 'data' / 'processed'


def main():
    ap = argparse.ArgumentParser(description='Run time-series model training')
    ap.add_argument('--ticker', required=True)
    args = ap.parse_args()

    path = PROC_DIR / f'features_{args.ticker.upper()}.parquet'
    if not path.exists():
        raise FileNotFoundError(f'Missing features file: {path}')

    df = pd.read_parquet(path)
    df = df.sort_values('date').reset_index(drop=True)

    feature_cols = [c for c in df.columns if c.startswith('sent_') or c.startswith('count_roll_') or c in ['daily_mean_sentiment','daily_count']]
    target_cls = 'target_dir_1d'
    target_reg = 'target_return_1d'

    X = df[feature_cols].fillna(0.0).values
    y_cls = df[target_cls].values
    y_reg = df[target_reg].values

    # Drop last row for training to avoid NaN target
    X_train = X[:-1]
    y_cls_train = y_cls[:-1]
    y_reg_train = y_reg[:-1]

    print(f'Training classifiers on {X_train.shape[0]} rows, {X_train.shape[1]} features')
    cls_results = train_classifiers(X_train, y_cls_train, models_config=None, n_splits=5)
    for name, res in cls_results.items():
        print(f"Classifier {name}: best CV f1={res['best_score']:.3f}")

    print(f'Training regressors on {X_train.shape[0]} rows, {X_train.shape[1]} features')
    reg_results = train_regressors(X_train, y_reg_train, models_config=None, n_splits=5)
    for name, res in reg_results.items():
        print(f"Regressor {name}: best CV negMSE={res['best_score']:.4f}")


if __name__ == '__main__':
    main()
