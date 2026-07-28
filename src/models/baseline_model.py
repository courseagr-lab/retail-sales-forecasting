import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from src.data.data_cleaning import load_config, project_path


def time_based_split(df, config):
    test_days = config["split"]["test_days"]
    split_point = len(df) - test_days
    train = df.iloc[:split_point].reset_index(drop=True)
    test = df.iloc[split_point:].reset_index(drop=True)
    return train, test

def naive_forecast(train, test, target):
    """Prediksi hari ke-t = nilai aktual 7 hari sebelumnya (seasonal naive, karena ada pola mingguan)."""
    combined = pd.concat([train, test], ignore_index=True)
    predictions = combined[target].shift(7).iloc[len(train):].values
    return predictions


def train_sarima(train, target, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7)):
    """seasonal_order period=7 karena pola mingguan yang sudah kita konfirmasi di EDA."""
    model = SARIMAX(
        train[target],
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    fitted = model.fit(disp=False)
    return fitted


def forecast_sarima(fitted_model, steps):
    forecast = fitted_model.forecast(steps=steps)
    return forecast.values