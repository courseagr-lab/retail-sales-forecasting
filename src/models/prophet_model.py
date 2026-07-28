import pandas as pd
from prophet import Prophet


def prepare_prophet_data(df, date_col, target):
    return df[[date_col, target]].rename(columns={date_col: "ds", target: "y"})


def train_prophet(train, date_col, target, config):
    prophet_train = prepare_prophet_data(train, date_col, target)

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
    )
    # tambahkan promo & holiday sebagai regressor eksternal (Prophet mendukung ini)
    model.add_regressor("promo")
    model.add_regressor("is_holiday")

    prophet_train["promo"] = train["promo"].values
    prophet_train["is_holiday"] = train["is_holiday"].values

    model.fit(prophet_train)
    return model


def forecast_prophet(model, test, date_col):
    future = test[[date_col]].rename(columns={date_col: "ds"})
    future["promo"] = test["promo"].values
    future["is_holiday"] = test["is_holiday"].values

    forecast = model.predict(future)
    return forecast["yhat"].values