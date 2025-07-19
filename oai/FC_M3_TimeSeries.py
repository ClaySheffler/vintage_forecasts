"""
Loss Curve Forecasting Using Time Series Modeling (Method #3)
--------------------------------------------------------------

This script implements Method #3: forecasting cumulative loss curves using a hierarchical or univariate
time series approach. It models each vintage's curve as a time series (MOB axis) and forecasts future
points using ARIMA.

Suitable for:
-------------
- Loss curves with consistent temporal structure
- Forecasting partial/incomplete vintages
- Incorporating uncertainty and smoothing

Expected Input:
---------------
A pandas DataFrame `df` with the following columns:
- 'vintage': str or int, vintage identifier (e.g., '2019Q2')
- 'mob': int, month-on-book (0, 1, ..., n)
- 'cum_loss': float, cumulative loss

This script assumes the DataFrame is already loaded into memory as `df`.

Dependencies:
-------------
- pandas, numpy, matplotlib, statsmodels
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from collections import defaultdict

# --- Forecast a single vintage using ARIMA ---
def forecast_vintage_arima(loss_series, max_forecast_mob=60):
    known_mobs = len(loss_series)
    steps_ahead = max_forecast_mob - known_mobs

    try:
        model = ARIMA(loss_series, order=(1, 1, 0))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=steps_ahead)
        full_series = np.concatenate([loss_series, forecast])
    except:
        full_series = np.concatenate([loss_series, [np.nan] * steps_ahead])

    return full_series

# --- Forecast all vintages ---
def forecast_all_vintages(df, max_forecast_mob=60, min_obs=6):
    results = {}
    df_sorted = df.sort_values(['vintage', 'mob'])
    vintages = df['vintage'].unique()

    for vintage in vintages:
        mob_series = df_sorted[df_sorted['vintage'] == vintage].sort_values('mob')['cum_loss'].values
        if len(mob_series) >= min_obs:
            full_series = forecast_vintage_arima(mob_series, max_forecast_mob)
            results[vintage] = full_series

    return pd.DataFrame(results).T

# --- Plot actual vs forecasted curves ---
def plot_forecasted_curves(df, df_forecast):
    plt.figure(figsize=(12, 6))
    for vintage in df_forecast.index:
        actual = df[df['vintage'] == vintage].sort_values('mob')['cum_loss'].values
        forecast = df_forecast.loc[vintage].values
        mob_range = np.arange(len(forecast))

        plt.plot(mob_range[:len(actual)], actual, linestyle='-', label=f"{vintage} actual")
        plt.plot(mob_range[len(actual):], forecast[len(actual):], linestyle='--', label=f"{vintage} forecast")

    plt.title("Time Series Forecast of Loss Curves (ARIMA)")
    plt.xlabel("Month on Book (MOB)")
    plt.ylabel("Cumulative Loss")
    plt.grid(True)
    plt.legend(loc='upper left', fontsize=8)
    plt.tight_layout()
    plt.show()

# --- Main Execution Block ---
if __name__ == "__main__":
    # Assume df is already loaded externally
    # df = pd.read_csv("loss_data.csv")

    # Forecast all vintages
    df_forecast = forecast_all_vintages(df, max_forecast_mob=60, min_obs=6)

    # Plot results
    plot_forecasted_curves(df, df_forecast)
