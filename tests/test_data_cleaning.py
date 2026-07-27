from pathlib import Path

import pandas as pd

from src.data.data_cleaning import (
    clean_sales_format,
    fill_date_gaps,
    handle_missing,
    load_config,
    load_raw,
    remove_duplicate_dates,
    run_cleaning,
)


def test_load_config_returns_expected_sections():
    config = load_config()

    raw_path = Path(config["data"]["raw_path"])

    assert raw_path.is_absolute()
    assert raw_path.name == "daily_sales_raw.csv"
    assert raw_path.exists()
    assert config["cleaning"]["outlier_rolling_window"] == 30
    assert config["cleaning"]["outlier_std_threshold"] == 3
    assert config["split"]["test_day"] == 90


def test_load_config_accepts_config_filename():
    config = load_config("config.yaml")

    assert config["data"]["target"] == "sales"


def test_load_raw_reads_dataset():
    config = load_config()

    df = load_raw(config)

    assert df.shape[0] == 1096
    assert pd.api.types.is_datetime64_any_dtype(df[config["data"]["date_col"]])


def test_cleaning_pipeline_helpers():
    config = load_config()
    date_col = config["data"]["date_col"]
    target = config["data"]["target"]
    df = pd.DataFrame(
        {
            date_col: pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-03"]),
            target: ["1,000", "500", None],
            "promo": [0, 1, 0],
            "is_holiday": [0, 0, 1],
        }
    )

    df = clean_sales_format(df, config)
    df = remove_duplicate_dates(df, config)
    df = fill_date_gaps(df, config)
    df = handle_missing(df, config)

    assert df.shape[0] == 3
    assert df.loc[0, target] == 750
    assert df.loc[1, "promo"] == 0
    assert not df[target].isna().any()


def test_run_cleaning_writes_interim_file():
    config = load_config()

    df = run_cleaning(config)

    assert Path(config["data"]["interim_path"]).exists()
    assert not df[config["data"]["target"]].isna().any()
