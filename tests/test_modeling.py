import numpy as np
import pandas as pd

from modeling.modeling import create_time_series_splits, train_classifiers, train_regressors


def test_time_series_split_increasing_windows():
    X = np.arange(50).reshape(-1, 1)
    tscv = create_time_series_splits(X, n_splits=5)
    lengths = []
    for train_idx, test_idx in tscv.split(X):
        lengths.append((len(train_idx), len(test_idx)))
        assert max(train_idx) < min(test_idx)  # no leakage
    assert len(lengths) == 5


def test_tiny_integration_trainers():
    # Synthetic data: 80 points, two simple features
    n = 80
    rng = np.random.default_rng(0)
    x1 = rng.normal(size=n).cumsum()
    x2 = rng.normal(size=n)
    ret = 0.1 * np.concatenate([[0], np.diff(x1)]) + 0.01 * x2 + rng.normal(scale=0.05, size=n)
    direction = (ret > 0).astype(int)

    X = np.column_stack([
        pd.Series(x1).rolling(3, min_periods=1).mean(),
        pd.Series(x2).rolling(5, min_periods=1).mean(),
    ])

    # Drop last row for target shift scenario
    X_train = X[:-1]
    y_cls = direction[:-1]
    y_reg = ret[:-1]

    cls = train_classifiers(X_train, y_cls, n_splits=3)
    reg = train_regressors(X_train, y_reg, n_splits=3)

    # Expect models to return dict with best_estimator and score
    assert all('best_estimator' in v for v in cls.values())
    assert all('best_estimator' in v for v in reg.values())
