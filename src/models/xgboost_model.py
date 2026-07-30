import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit

from src.data.data_cleaning import load_config, project_path
from src.models.baseline_model import time_based_split


FEATURE_COLS_EXCLUDE = ["date", "sales"]  # kolom yang bukan fitur


def get_feature_columns(df, target, exclude=("date",)):
    return [c for c in df.columns if c != target and c not in exclude]


def prepare_xy(df, target, feature_cols):
    X = df[feature_cols]
    y = df[target]
    return X, y


def objective(trial, X_train, y_train, config):
    """Objective function untuk Optuna — dipanggil berulang, tiap kali coba kombinasi hyperparameter berbeda."""
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "random_state": config["model"]["random_state"],
    }

    # TimeSeriesSplit -> WAJIB untuk time series, bukan KFold biasa
    # tiap fold: training selalu dari masa lalu, validasi dari "masa depan" relatif ke fold itu
    tscv = TimeSeriesSplit(n_splits=5)
    rmse_scores = []

    for train_idx, val_idx in tscv.split(X_train):
        X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        model = xgb.XGBRegressor(**params)
        model.fit(X_fold_train, y_fold_train)
        preds = model.predict(X_fold_val)

        rmse = np.sqrt(np.mean((y_fold_val.values - preds) ** 2))
        rmse_scores.append(rmse)

    return np.mean(rmse_scores)


def tune_xgboost(X_train, y_train, config, n_trials=50):
    study = optuna.create_study(direction="minimize")
    study.optimize(
        lambda trial: objective(trial, X_train, y_train, config),
        n_trials=n_trials,
        show_progress_bar=True,
    )
    return study


def train_final_xgboost(X_train, y_train, best_params, config):
    params = {**best_params, "random_state": config["model"]["random_state"]}
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train)
    return model