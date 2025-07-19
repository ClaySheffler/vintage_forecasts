# Cumulative Gross Charge-Off Forecast Using Weibull Curve

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import weibull_min

# --- Parameters and Data Generation ---
np.random.seed(7)
segments = ["Segment_A", "Segment_B", "Segment_C"]
vintage_years = list(range(2015, 2025))
term_months = 144
delay_months = 6

segment_params = {
    seg: {
        "shape": np.random.uniform(1.4, 2.0),
        "scale": np.random.uniform(35, 60)
    } for seg in segments
}

data = []
for seg in segments:
    for yr in vintage_years:
        max_gco = np.random.uniform(0.05, 0.15) * 0.9 ** (yr - 2015)
        months = np.arange(term_months)
        c, scale = segment_params[seg]["shape"], segment_params[seg]["scale"]
        cdf = weibull_min.cdf(months - delay_months, c=c, scale=scale)
        cdf[months < delay_months] = 0
        cumulative_gco = max_gco * cdf
        for mob, gco in enumerate(cumulative_gco):
            data.append({"Segment": seg, "Vintage": pd.Timestamp(f"{yr}-01-01"), "MOB": mob, "Cumulative_GCO": gco})

df_all = pd.DataFrame(data)

# --- Model Fitting on Early Vintages ---
train_years = [2015, 2016, 2017]
forecast_years = [2019, 2020, 2021, 2022, 2023, 2024]
train_df = df_all[df_all["Vintage"].dt.year.isin(train_years)]
forecast_df = df_all[df_all["Vintage"].dt.year.isin(forecast_years)]

def weibull_cdf(x, shape, scale):
    return weibull_min.cdf(x, c=shape, scale=scale)

fit_results = {}
for seg in segments:
    seg_train = train_df[train_df["Segment"] == seg].copy()
    seg_train = seg_train[seg_train["MOB"] >= delay_months]
    xdata = seg_train["MOB"] - delay_months
    ydata = seg_train["Cumulative_GCO"] / seg_train.groupby(["Segment", "Vintage"])["Cumulative_GCO"].transform("max")
    popt, _ = curve_fit(weibull_cdf, xdata, ydata, p0=[1.5, 50], bounds=([0.8, 10], [4.0, 150]))
    fit_results[seg] = {"shape": popt[0], "scale": popt[1]}

# --- Forecasting for Later Vintages ---
forecast_records = []
for seg in segments:
    shape_hat, scale_hat = fit_results[seg]["shape"], fit_results[seg]["scale"]
    for yr in forecast_years:
        vint_df = forecast_df[(forecast_df["Segment"] == seg) & (forecast_df["Vintage"].dt.year == yr)].copy()
        obs_months = 36
        obs_df = vint_df[vint_df["MOB"] <= obs_months]
        mob_obs = obs_df["MOB"].values
        gco_obs = obs_df["Cumulative_GCO"].values
        last_mob = mob_obs[-1]
        last_adj = max(last_mob - delay_months, 1)
        model_cdf_last = weibull_cdf(last_adj, shape_hat, scale_hat)
        amplitude = gco_obs[-1] / model_cdf_last if model_cdf_last > 0 else gco_obs[-1]
        months = vint_df["MOB"].values
        mob_adj = np.maximum(months - delay_months, 0)
        model_cdf_full = weibull_cdf(mob_adj, shape_hat, scale_hat)
        gco_hat_full = amplitude * model_cdf_full
        vint_df = vint_df.assign(Forecast=gco_hat_full)
        forecast_records.append(vint_df)

df_forecast = pd.concat(forecast_records)
df_forecast["Vintage_Date"] = df_forecast["Vintage"]
df_forecast["Months_On_Book"] = ((pd.Timestamp("2025-07-01") - df_forecast["Vintage_Date"]) / np.timedelta64(1, 'M')).astype(int)
df_forecast["Observed"] = df_forecast["MOB"] <= df_forecast["Months_On_Book"]

# --- Final Plot: Unified Chart ---
sample_vintages = df_forecast.groupby(["Segment", "Vintage"]).head(1).sample(6, random_state=3)[["Segment", "Vintage"]]
fig, ax = plt.subplots(figsize=(12, 7))
for seg, vint in sample_vintages.values:
    tmp = df_forecast[(df_forecast["Segment"] == seg) & (df_forecast["Vintage"] == vint)].copy()
    obs_limit = tmp["Months_On_Book"].iloc[0]
    actuals = tmp[tmp["MOB"] <= obs_limit]
    future = tmp[tmp["MOB"] > obs_limit]
    label = f"{seg} | {vint.year}"
    ax.plot(actuals["MOB"], actuals["Cumulative_GCO"], lw=2, label=f"{label} (Actual)")
    ax.plot(future["MOB"], future["Forecast"], lw=2, linestyle="--", label=f"{label} (Forecast)")

ax.set_title("Cumulative GCO: Actuals (solid) and Forecasts (dashed) as of July 2025", fontsize=14)
ax.set_xlabel("MOB")
ax.set_ylabel("Cumulative GCO")
ax.legend(ncol=2, fontsize=9)
ax.grid(True)
plt.tight_layout()
plt.show()
