from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Generator, Iterable, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.base import is_classifier, is_regressor
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, VotingClassifier, VotingRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
try:
    from imblearn.over_sampling import SMOTE
    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.pipeline import Pipeline as ImbPipeline
    IMBLEARN_AVAILABLE = True
except ImportError:
    SMOTE = None
    RandomUnderSampler = None
    ImbPipeline = None
    IMBLEARN_AVAILABLE = False

try:
    from xgboost import XGBClassifier, XGBRegressor  # type: ignore
except Exception:  # pragma: no cover
    XGBClassifier = None  # type: ignore
    XGBRegressor = None  # type: ignore

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def create_time_series_splits(X: pd.DataFrame | np.ndarray, y: Optional[pd.Series | np.ndarray] = None, n_splits: int = 5) -> TimeSeriesSplit:
    """Create a TimeSeriesSplit object configured for no shuffling and expanding window.

    Returns the splitter to be used with GridSearchCV or manual iteration.
    """
    return TimeSeriesSplit(n_splits=n_splits)


def _ensure_2d(X: pd.DataFrame | np.ndarray) -> np.ndarray:
    if isinstance(X, pd.DataFrame):
        return X.values
    X = np.asarray(X)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    return X


def train_classifiers(
    X_train: pd.DataFrame | np.ndarray,
    y_train: pd.Series | np.ndarray,
    models_config: Optional[Dict[str, Dict]] = None,
    n_splits: int = 5,
) -> Dict[str, Dict]:
    """Train several classifiers with TimeSeriesSplit GridSearch.

    Returns dict mapping model name -> {best_estimator, best_score, cv_results}
    """
    X_train = _ensure_2d(X_train)
    y_train = np.asarray(y_train)

    tscv = create_time_series_splits(X_train, y_train, n_splits=n_splits)

    results: Dict[str, Dict] = {}

    default_configs: Dict[str, Dict] = {
        "logreg": {
            "pipeline": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=200, n_jobs=None)),
            ]),
            "param_grid": {
                "clf__C": [0.1, 1.0, 5.0],
                "clf__penalty": ["l2"],
                "clf__solver": ["lbfgs"],
            },
        },
        "rf": {
            "pipeline": Pipeline([
                ("clf", RandomForestClassifier(n_estimators=200, random_state=42))
            ]),
            "param_grid": {
                "clf__n_estimators": [100, 200],
                "clf__max_depth": [None, 5, 10],
                "clf__min_samples_leaf": [1, 3],
            },
        },
    }

    if XGBClassifier is not None:
        default_configs["xgb"] = {
            "pipeline": Pipeline([
                ("clf", XGBClassifier(
                    n_estimators=200,
                    max_depth=3,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    objective="binary:logistic",
                    eval_metric="logloss",
                    tree_method="hist",
                    random_state=42,
                ))
            ]),
            "param_grid": {
                "clf__n_estimators": [100, 200],
                "clf__max_depth": [2, 3, 4],
                "clf__learning_rate": [0.05, 0.1],
            },
        }

    cfgs = models_config or default_configs

    for name, cfg in cfgs.items():
        pipe: Pipeline = cfg["pipeline"]
        grid = cfg.get("param_grid", {})
        gs = GridSearchCV(pipe, grid, cv=tscv, scoring="f1", n_jobs=1, refit=True)
        gs.fit(X_train, y_train)
        results[name] = {
            "best_estimator": gs.best_estimator_,
            "best_score": gs.best_score_,
            "cv_results": gs.cv_results_,
        }
        # Save
        joblib.dump(gs.best_estimator_, MODELS_DIR / f"clf_{name}.joblib")
        
        # Save feature names for prediction
        if hasattr(X_train, 'columns'):
            feature_names = list(X_train.columns)
        else:
            feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]
        
        with open(MODELS_DIR / "feature_names.json", "w") as f:
            json.dump(feature_names, f)

    # Create ensemble voting classifier if we have multiple models
    if len(results) > 1:
        print("Creating ensemble voting classifier...")
        raw_estimators = [(name, result["best_estimator"]) for name, result in results.items()]

        def _is_classifier_estimator(est) -> bool:
            if is_classifier(est):
                return True
            if hasattr(est, "steps") and len(getattr(est, "steps", [])) > 0:
                return is_classifier(est.steps[-1][1])
            return False

        estimators = [(name, est) for name, est in raw_estimators if _is_classifier_estimator(est)]
        if len(estimators) < 2:
            print("Skipping ensemble: fewer than two valid classifiers.")
            return results

        # Use soft voting only if all have predict_proba; otherwise fall back to hard
        all_have_proba = all(hasattr(est, "predict_proba") or (hasattr(est, "steps") and hasattr(est.steps[-1][1], "predict_proba")) for _, est in estimators)
        voting_mode = 'soft' if all_have_proba else 'hard'
        ensemble = VotingClassifier(estimators=estimators, voting=voting_mode)
        ensemble.fit(X_train, y_train)
        
        # Evaluate ensemble
        ensemble_score = ensemble.score(X_train, y_train)
        results['ensemble'] = {
            "best_estimator": ensemble,
            "best_score": ensemble_score,
            "cv_results": None,
        }
        joblib.dump(ensemble, MODELS_DIR / "clf_ensemble.joblib")
        print(f"Ensemble accuracy: {ensemble_score:.4f}")

    return results


def train_regressors(
    X_train: pd.DataFrame | np.ndarray,
    y_train: pd.Series | np.ndarray,
    models_config: Optional[Dict[str, Dict]] = None,
    n_splits: int = 5,
) -> Dict[str, Dict]:
    """Train regressors with TimeSeriesSplit GridSearch.

    Returns dict mapping model name -> {best_estimator, best_score, cv_results}
    """
    X_train = _ensure_2d(X_train)
    y_train = np.asarray(y_train)

    tscv = create_time_series_splits(X_train, y_train, n_splits=n_splits)
    results: Dict[str, Dict] = {}

    default_configs: Dict[str, Dict] = {
        "ridge": {
            "pipeline": Pipeline([
                ("scaler", StandardScaler()),
                ("reg", Ridge())
            ]),
            "param_grid": {
                "reg__alpha": [0.1, 1.0, 5.0],
            },
        },
        "rfr": {
            "pipeline": Pipeline([
                ("reg", RandomForestRegressor(n_estimators=200, random_state=42))
            ]),
            "param_grid": {
                "reg__n_estimators": [100, 200],
                "reg__max_depth": [None, 5, 10],
                "reg__min_samples_leaf": [1, 3],
            },
        },
    }

    if XGBRegressor is not None:
        default_configs["xgbr"] = {
            "pipeline": Pipeline([
                ("reg", XGBRegressor(
                    n_estimators=200,
                    max_depth=3,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    objective="reg:squarederror",
                    tree_method="hist",
                    random_state=42,
                ))
            ]),
            "param_grid": {
                "reg__n_estimators": [100, 200],
                "reg__max_depth": [2, 3, 4],
                "reg__learning_rate": [0.05, 0.1],
            },
        }

    cfgs = models_config or default_configs

    for name, cfg in cfgs.items():
        pipe: Pipeline = cfg["pipeline"]
        grid = cfg.get("param_grid", {})
        gs = GridSearchCV(pipe, grid, cv=tscv, scoring="neg_mean_squared_error", n_jobs=1, refit=True)
        gs.fit(X_train, y_train)
        results[name] = {
            "best_estimator": gs.best_estimator_,
            "best_score": gs.best_score_,
            "cv_results": gs.cv_results_,
        }
        joblib.dump(gs.best_estimator_, MODELS_DIR / f"reg_{name}.joblib")
        
        # Save feature names for prediction (same as classifier)
        if hasattr(X_train, 'columns'):
            feature_names = list(X_train.columns)
        else:
            feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]
        
        with open(MODELS_DIR / "feature_names.json", "w") as f:
            json.dump(feature_names, f)

    # Create ensemble voting regressor if we have multiple models
    if len(results) > 1:
        print("Creating ensemble voting regressor...")
        raw_estimators = [(name, result["best_estimator"]) for name, result in results.items()]
        def _is_regressor_estimator(est) -> bool:
            if is_regressor(est):
                return True
            if hasattr(est, "steps") and len(getattr(est, "steps", [])) > 0:
                return is_regressor(est.steps[-1][1])
            return False
        estimators = [(name, est) for name, est in raw_estimators if _is_regressor_estimator(est)]
        if len(estimators) < 2:
            print("Skipping ensemble regressor: fewer than two valid regressors.")
            return results
        ensemble = VotingRegressor(estimators=estimators)
        ensemble.fit(X_train, y_train)
        
        # Evaluate ensemble
        ensemble_score = ensemble.score(X_train, y_train)
        results['ensemble'] = {
            "best_estimator": ensemble,
            "best_score": ensemble_score,
            "cv_results": None,
        }
        joblib.dump(ensemble, MODELS_DIR / "reg_ensemble.joblib")
        print(f"Ensemble R² score: {ensemble_score:.4f}")

    return results


def build_lstm_sequence_model(*args, **kwargs):  # pragma: no cover - optional stub
    """Optional: Keras LSTM model builder (not implemented to keep deps light).

    Documented approach: create sequences of length `lookback`, build a small LSTM
    for classification/regression, ensuring time-order splits. Requires TensorFlow/Keras.
    """
    raise NotImplementedError("LSTM model is optional and not implemented in this environment.")
