import pandas as pd
import numpy as np

from src.data.data_cleaning import load_config, project_path

def load_interim(config):
    df = pd.read_csv(config['data']['interim_path'])
    df['date'] = pd.to_datetime(df['date'])
    return df


def add_calendar_features(df):
    df = df.copy()
    df['day_of_week'] = df['date'].dt.dayofweek
    df['is_weekend'] = (df['day_of_week'] >=5).astype(int)
    df['month'] = df['date'].dt.month
    df['day_of_year'] = df['date'].dt.dayofyear
    df['week_of_year']= df['date'].dt.isocalendar().week.astype(int)
    return df

def add_lag_features(df, target, lags=(1, 7, 14, 30)):
    df = df.copy()
    for lag in lags:
        df[f"{target}_lag_{lag}"] = df[target].shift(lag)
    return df

def add_rolling_features(df, target, windows=(7,30)):
    df = df.copy()
    for window in windows:
         # shift(1) dulu supaya rolling window TIDAK termasuk hari ini sendiri (hindari data leakage)
         df[f"{target}_rolling_mean{window}"] = df[target].shift(1).rolling(window).mean()
         df[f"{target}_rolling_std{window}"] = df[target].shift(1).rolling(window).std()
    return df

def build_features(config):
    target = config['data']['target']
    df = load_interim(config)
    df = add_calendar_features(df)
    df = add_lag_features(df, target)
    df = add_rolling_features(df, target)

    # baris awal akan punya NaN karena lag/rolling butuh histori sebelum tanggal tsb.
    # ini WAJIB di-drop, bukan diimputasi -> tidak ada data histori untuk mengisinya
    df = df.dropna().reset_index(drop=True)

    df.to_csv(project_path(config["data"]["processed_path"]), index=False)
    return df

if __name__ == "__main__":
    config = load_config()
    df_processed = build_features(config)
    print(f"Processed shape: {df_processed.shape}")
    print(df_processed.columns.tolist())
    print(df_processed.head())