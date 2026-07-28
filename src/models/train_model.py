import mlflow
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pandas as pd
from src.data.data_cleaning import load_config, project_path
from src.models.baseline_model import time_based_split, naive_forecast, train_sarima, forecast_sarima
from src.models.prophet_model import train_prophet, forecast_prophet


def evaluate_forecast(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return {"mae": mae, "rmse": rmse, "mape": mape}


def setup_mlflow(config):
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])


def log_run(model_name, params, metrics, config):
    with mlflow.start_run(run_name=model_name):
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.set_tag("model_type", model_name)


def run_all_baselines(config):
    df = pd.read_csv(project_path(config["data"]["processed_path"]))
    df["date"] = pd.to_datetime(df["date"])
    target = config["data"]["target"]

    train, test = time_based_split(df, config)
    setup_mlflow(config)

    results = {}

    # Naive seasonal
    naive_preds = naive_forecast(train, test, target)
    naive_metrics = evaluate_forecast(test[target].values, naive_preds)
    log_run("naive_seasonal", {"lag": 7}, naive_metrics, config)
    results["naive_seasonal"] = naive_metrics

    # SARIMA
    sarima_order = (1, 1, 1)
    sarima_seasonal = (1, 1, 1, 7)
    sarima_model = train_sarima(train, target, order=sarima_order, seasonal_order=sarima_seasonal)
    sarima_preds = forecast_sarima(sarima_model, steps=len(test))
    sarima_metrics = evaluate_forecast(test[target].values, sarima_preds)
    log_run("sarima", {"order": sarima_order, "seasonal_order": sarima_seasonal}, sarima_metrics, config)
    results["sarima"] = sarima_metrics

    # Prophet
    prophet_model = train_prophet(train, "date", target, config)
    prophet_preds = forecast_prophet(prophet_model, test, "date")
    prophet_metrics = evaluate_forecast(test[target].values, prophet_preds)
    log_run("prophet", {"yearly_seasonality": True, "weekly_seasonality": True}, prophet_metrics, config)
    results["prophet"] = prophet_metrics

    return results


if __name__ == "__main__":
    import pandas as pd
    config = load_config()
    results = run_all_baselines(config)
    for model_name, metrics in results.items():
        print(f"{model_name}: {metrics}")