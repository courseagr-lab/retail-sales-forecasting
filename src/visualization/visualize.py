import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from src.data.data_cleaning import project_path


def save_fig(fig, name, figures_path):
    figures_dir = project_path(figures_path)
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures_dir / f"{name}.png", bbox_inches="tight", dpi=150)


def plot_timeseries(df, date_col, target, figures_path):
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(df[date_col], df[target])
    ax.set_title(f"{target} over time")
    save_fig(fig, "timeseries_overview", figures_path)
    plt.show()


def plot_decomposition(df, date_col, target, figures_path, period=365):
    ts = df.set_index(date_col)[target]
    result = seasonal_decompose(ts, model="additive", period=period)

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    result.observed.plot(ax=axes[0], title="Observed")
    result.trend.plot(ax=axes[1], title="Trend")
    result.seasonal.plot(ax=axes[2], title="Seasonal")
    result.resid.plot(ax=axes[3], title="Residual")
    plt.tight_layout()
    save_fig(fig, "decomposition", figures_path)
    plt.show()
    return result


def plot_weekly_pattern(df, date_col, target, figures_path):
    df = df.copy()
    df["day_of_week"] = df[date_col].dt.day_name()
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.boxplot(data=df, x="day_of_week", y=target, order=order, ax=ax)
    ax.set_title(f"{target} by Day of Week")
    plt.xticks(rotation=45)
    save_fig(fig, "weekly_pattern", figures_path)
    plt.show()


def plot_monthly_pattern(df, date_col, target, figures_path):
    df = df.copy()
    df["month"] = df[date_col].dt.month

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.boxplot(data=df, x="month", y=target, ax=ax)
    ax.set_title(f"{target} by Month")
    save_fig(fig, "monthly_pattern", figures_path)
    plt.show()


def plot_acf_pacf(df, target, figures_path, lags=30):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    plot_acf(df[target], lags=lags, ax=axes[0])
    plot_pacf(df[target], lags=lags, ax=axes[1])
    axes[0].set_title("Autocorrelation (ACF)")
    axes[1].set_title("Partial Autocorrelation (PACF)")
    plt.tight_layout()
    save_fig(fig, "acf_pacf", figures_path)
    plt.show()
