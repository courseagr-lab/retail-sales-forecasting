import yaml
from pathlib import Path
import pandas as pd
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PATH_KEYS = {
    "raw_path",
    "interim_path",
    "processed_path",
    "model_path",
    "metrics_path",
    "figures_path",
}


def project_path(path):
    path = Path(path)
    if path.is_absolute():
        return path

    candidate = PROJECT_ROOT / path
    if candidate.exists():
        return candidate

    config_candidate = PROJECT_ROOT / "config" / path
    if path.name == "config.yaml" and config_candidate.exists():
        return config_candidate

    return candidate


def resolve_config_paths(config):
    for section in config.values():
        if not isinstance(section, dict):
            continue

        for key, value in section.items():
            if key in PATH_KEYS and isinstance(value, str):
                section[key] = str(project_path(value))

    return config


def load_config(path="config/config.yaml"):
    """
    Load the configuration file.

    Args:
        path (str): Path to the configuration file.

    Returns:
        dict: Configuration parameters.
    """
    with open(project_path(path), "r") as file:
        config = yaml.safe_load(file)
    return resolve_config_paths(config)


def load_raw(config):
    df = pd.read_csv(project_path(config["data"]["raw_path"]))
    df[config["data"]["date_col"]] = pd.to_datetime(df[config["data"]["date_col"]])
    return df


def clean_sales_format(df, config):
    df = df.copy()
    target = config["data"]["target"]
    df[target] = df[target].astype(str).str.replace(",", "", regex=False)
    df[target] = pd.to_numeric(df[target], errors="coerce")
    return df


def remove_duplicate_dates(df, config):
    df = df.copy()
    date_col = config["data"]["date_col"]
    target = config["data"]["target"]
    df = df.groupby(date_col, as_index=False).agg(
        {
            target: "mean",
            "promo": "max",
            "is_holiday": "max",
        }
    )
    return df


def fill_date_gaps(df, config):
    df = df.copy()
    date_col = config["data"]["date_col"]
    df = df.set_index(date_col)

    full_range = pd.date_range(df.index.min(), df.index.max(), freq="D")
    df = df.reindex(full_range)
    df.index.name = date_col

    df["promo"] = df["promo"].fillna(0).astype(int)
    df["is_holiday"] = df["is_holiday"].fillna(0).astype(int)
    return df.reset_index()


def handle_outliers(df, config):
    df = df.copy()
    target = config["data"]["target"]
    cleaning_config = config["cleaning"]

    rolling_median = (
        df[target]
        .rolling(
            cleaning_config["outlier_rolling_window"],
            center=True,
            min_periods=1,
        )
        .median()
    )
    rolling_std = (
        df[target]
        .rolling(
            cleaning_config["outlier_rolling_window"],
            center=True,
            min_periods=1,
        )
        .std()
    )

    is_outlier = (df[target] - rolling_median).abs() > (
        cleaning_config["outlier_std_threshold"] * rolling_std
    )
    df.loc[is_outlier, target] = np.nan
    return df


def handle_missing(df, config):
    df = df.copy()
    target = config["data"]["target"]
    df[target] = df[target].interpolate(method="linear", limit_direction="both")
    return df


def run_cleaning(config):
    df = load_raw(config)
    df = clean_sales_format(df, config)
    df = remove_duplicate_dates(df, config)
    df = fill_date_gaps(df, config)
    df = handle_outliers(df, config)
    df = handle_missing(df, config)
    df = df.sort_values(config["data"]["date_col"]).reset_index(drop=True)

    interim_path = project_path(config["data"]["interim_path"])
    interim_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(interim_path, index=False)
    return df

if __name__ == "__main__":
    config = load_config()
    df_clean = run_cleaning(config)
    print(f"Cleaned shape: {df_clean.shape}")
    print(f"Missing values:\n{df_clean.isnull().sum()}")
    print(f"Date range: {df_clean['date'].min()} to {df_clean['date'].max()}")
    print(f"Expected days: {(df_clean['date'].max() - df_clean['date'].min()).days + 1}")