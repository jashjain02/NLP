import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge

from evaluation.evaluate import eval_classifier, eval_regressor


def _make_features(tmp_path: Path) -> pd.DataFrame:
    n = 40
    dates = pd.date_range('2024-01-01', periods=n, freq='B')
    rng = np.random.default_rng(0)
    sent = rng.normal(0, 0.1, size=n)
    ret = rng.normal(0, 0.01, size=n)
    df = pd.DataFrame({
        'date': dates,
        'daily_mean_sentiment': pd.Series(sent).rolling(3, min_periods=1).mean(),
        'daily_count': np.random.randint(0, 5, size=n),
        'return': ret,
        'target_return_1d': pd.Series(ret).shift(-1),
    })
    df['target_dir_1d'] = (df['target_return_1d'] > 0).astype(int)
    return df


def test_eval_classifier_and_regressor(tmp_path: Path, monkeypatch):
    df = _make_features(tmp_path)

    X = df[['daily_mean_sentiment','daily_count']].fillna(0.0).values
    y_cls = df['target_dir_1d'].fillna(0).astype(int).values
    y_reg = df['target_return_1d'].fillna(0.0).values

    clf = Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(max_iter=200))])
    clf.fit(X[:-1], y_cls[:-1])

    reg = Pipeline([('scaler', StandardScaler()), ('reg', Ridge())])
    reg.fit(X[:-1], y_reg[:-1])

    clf_path = tmp_path / 'tmp_clf.joblib'
    reg_path = tmp_path / 'tmp_reg.joblib'
    joblib.dump(clf, clf_path)
    joblib.dump(reg, reg_path)

    # Monkeypatch data load to return our df
    from evaluation import evaluate as ev
    def _fake_load_features(ticker: str):
        return df
    monkeypatch.setattr(ev, '_load_features', _fake_load_features)

    metrics_clf = ev.eval_classifier(clf_path, df, 'TEST')
    metrics_reg = ev.eval_regressor(reg_path, df, 'TEST')

    assert 'report' in metrics_clf
    assert 'mse' in metrics_reg
