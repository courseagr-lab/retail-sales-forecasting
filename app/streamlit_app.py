import sys
from pathlib import Path

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data.data_cleaning import load_config, project_path
from src.models.baseline_model import time_based_split
from src.models.prophet_model import load_prophet_model, prepare_prophet_data
from src.models.train_model import evaluate_forecast


st.set_page_config(page_title="Retail Sales Forecast", layout="wide")
st.title("Retail Sales Forecasting Dashboard")

config = load_config("config/config.yaml")

@st.cache_data
def load_data():
    df = pd.read_csv(project_path(config["data"]["processed_path"]))
    df["date"] = pd.to_datetime(df["date"])
    return df

@st.cache_resource
def get_model():
    return load_prophet_model(config)

df = load_data()
model = get_model()
target = config["data"]["target"]

train, test = time_based_split(df, config)

# Forecast pada test period
future = test[["date"]].rename(columns={"date": "ds"})
future["promo"] = test["promo"].values
future["is_holiday"] = test["is_holiday"].values
forecast = model.predict(future)

metrics = evaluate_forecast(test[target].values, forecast["yhat"].values)

# --- Sidebar ---
st.sidebar.header("Info Model")
st.sidebar.metric("MAE", f"{metrics['mae']:.2f}")
st.sidebar.metric("RMSE", f"{metrics['rmse']:.2f}")
st.sidebar.metric("MAPE", f"{metrics['mape']:.2f}%")

st.sidebar.markdown("---")
show_components = st.sidebar.checkbox("Tampilkan komponen forecast (trend/seasonality)", value=False)

# --- Main chart ---
st.subheader("Actual vs Forecast (90 Hari Terakhir)")

fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(train["date"].iloc[-30:], train[target].iloc[-30:], label="Train (30 hari terakhir)", color="gray")
ax.plot(test["date"], test[target].values, label="Actual", color="black")
ax.plot(test["date"], forecast["yhat"].values, label="Prophet Forecast", linestyle="--", color="green")
ax.fill_between(
    test["date"], forecast["yhat_lower"], forecast["yhat_upper"],
    alpha=0.2, color="green", label="Confidence Interval",
)
ax.legend()
ax.set_xlabel("Tanggal")
ax.set_ylabel("Sales")
st.pyplot(fig)

# --- Komponen (opsional) ---
if show_components:
    st.subheader("Komponen Forecast")
    fig2 = model.plot_components(forecast)
    st.pyplot(fig2)

# --- Tabel data ---
st.subheader("Detail Prediksi")
result_table = test[["date", target]].copy()
result_table["forecast"] = forecast["yhat"].values
result_table["error"] = result_table[target] - result_table["forecast"]
st.dataframe(result_table, use_container_width=True)

# --- Input interaktif: prediksi hari baru ---
st.subheader("Coba Prediksi Hari Tertentu")
col1, col2, col3 = st.columns(3)
with col1:
    input_date = st.date_input("Tanggal", value=test["date"].max())
with col2:
    input_promo = st.selectbox("Promo?", [0, 1])
with col3:
    input_holiday = st.selectbox("Hari libur?", [0, 1])

if st.button("Prediksi"):
    single_future = pd.DataFrame({
        "ds": [pd.Timestamp(input_date)],
        "promo": [input_promo],
        "is_holiday": [input_holiday],
    })
    single_forecast = model.predict(single_future)
    predicted_sales = single_forecast["yhat"].values[0]
    st.success(f"Prediksi penjualan: **{predicted_sales:,.1f}**")