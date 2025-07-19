# filename: weibull_forecast_visualizer.py

import marimo
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import weibull_min

__generated_with = "0.4.1"
app = marimo.App()


@app.cell
def __():
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import weibull_min
    return np, plt, weibull_min


@app.cell
def __():
    app.title("Weibull-Based Charge-Off Forecasting")
    return


@app.cell
def __(mo):
    k = mo.slider(0.5, 5.0, value=1.5, step=0.1, label="Shape (k): Hazard Behavior")
    lam = mo.slider(3.0, 60.0, value=18.0, step=1.0, label="Scale (λ): Charge-Off Timing")
    t_obs = mo.slider(1, 60, value=12, step=1, label="Observation Month (T_obs)")
    co_obs = mo.slider(0.0, 0.5, value=0.03, step=0.005, label="Observed Charge-Off Rate (up to T_obs)")
    horizon = mo.slider(12, 60, value=60, step=1, label="Forecast Horizon (months)")
    return k, lam, t_obs, co_obs, horizon
__dependencies__ = ["mo"]


@app.cell
def __(np, horizon):
    x = np.linspace(0, horizon.value, 500)
    return x,


@app.cell
def __(weibull_min, x, k, lam):
    cdf_full = weibull_min.cdf(x, c=k.value, scale=lam.value)
    return cdf_full,
  

@app.cell
def __(weibull_min, t_obs, co_obs, k, lam):
    # Scale full CDF to match observed charge-offs
    cdf_obs = weibull_min.cdf(t_obs.value, c=k.value, scale=lam.value)
    if cdf_obs > 0:
        estimated_total_loss = co_obs.value / cdf_obs
    else:
        estimated_total_loss = 0.0
    return cdf_obs, estimated_total_loss
  

@app.cell
def __(x, t_obs, co_obs, cdf_full, estimated_total_loss):
    # Forecast curve
    cumulative_loss = estimated_total_loss * cdf_full

    # Separate actuals (solid) and forecast (dotted)
    x_actual = x[x <= t_obs.value]
    y_actual = cumulative_loss[x <= t_obs.value]

    x_forecast = x[x > t_obs.value]
    y_forecast = cumulative_loss[x > t_obs.value]

    return x_actual, y_actual, x_forecast, y_forecast, cumulative_loss
  

@app.cell
def __(plt, x_actual, y_actual, x_forecast, y_forecast, t_obs, co_obs, estimated_total_loss):
    fig, ax = plt.subplots(figsize=(10, 5))

    # Actual (solid line)
    ax.plot(x_actual, y_actual, label="Actual", color="black", linewidth=2)

    # Forecast (dotted line)
    ax.plot(x_forecast, y_forecast, label="Forecast", color="steelblue", linestyle="--", linewidth=2)

    # Marker for observed data point
    ax.scatter([t_obs.value], [co_obs.value], color="red", zorder=5)
    ax.annotate(f"Obs: {co_obs.value*100:.1f}%", (t_obs.value, co_obs.value),
                textcoords="offset points", xytext=(5, 5), ha='left', color="red")

    # Final point annotation
    final_forecast = y_forecast[-1] if len(y_forecast) > 0 else y_actual[-1]
    ax.axhline(final_forecast, color="gray", linestyle=":", linewidth=1)
    ax.annotate(f"Forecasted Total: {final_forecast*100:.2f}%",
                (x_forecast[-1] if len(x_forecast) else x_actual[-1], final_forecast),
                textcoords="offset points", xytext=(5, -15), ha='left', fontsize=9)

    ax.set_title("Cumulative Charge-Off Forecast (Actual + Weibull Projection)")
    ax.set_xlabel("Vintage Age (Months)")
    ax.set_ylabel("Cumulative Charge-Off Rate")
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    plt.show()
    return fig, ax
