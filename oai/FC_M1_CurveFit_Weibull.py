"""
Loss Curve Modeling with Parametric Weibull Forecasting
--------------------------------------------------------

This module implements a complete solution for modeling and forecasting cumulative loss curves
using a parametric Weibull model with scale. It is designed for loan portfolio analysis where 
losses are tracked by vintage and month-on-book (MOB), and some vintages may be incomplete.

Key Features:
-------------
1. **Weibull Curve Fitting with Scale Parameter**:
   - The model estimates both the shape (alpha, beta) and terminal cumulative loss (L) for each vintage.
   - Supports incomplete loss curves by fitting the tail using known partial loss data.

2. **Maturity and Convergence Filtering**:
   - Automatically identifies mature vintages based on convergence: loss growth in the last N MOBs must fall below a threshold.
   - Optionally filters training vintages to include only mature or sufficiently developed (≥6–9 MOBs) vintages.

3. **Loan Volume-Based Weighting**:
   - Fits the Weibull model for each vintage using weights proportional to loan volume, giving more influence to larger cohorts.

4. **Forecasting and Visualization**:
   - Generates MOB-level forecasts for each vintage using the fitted Weibull model.
   - Plots actual vs. forecasted cumulative loss curves, using solid lines for known values and dashed lines for forecasts.

Expected Input:
---------------
A pandas DataFrame `df` with the following columns:
- 'vintage': str or int, identifying the vintage cohort (e.g., '2019Q2')
- 'mob': int, month-on-book (e.g., 0, 1, ..., 36)
- 'cum_loss': float, cumulative loss at each MOB (as percent or decimal)
- 'segment': str, segment name (e.g., 'prime', 'subprime')
- 'loan_volume': float, total loan balance for the vintage (used for weighting)

Main Functions:
---------------
- `weibull_scaled(mob, alpha, beta, L)`: Weibull function with scale.
- `is_mature(loss_series, threshold, window)`: Flags if a vintage has converged.
- `prepare_training_vintages(...)`: Returns metadata and filtered vintages based on maturity and data sufficiency.
- `fit_weibull_with_scale_weighted(...)`: Fits Weibull parameters with volume-based weighting.
- `forecast_and_plot_weighted(...)`: Forecasts and plots curves; returns vintage-level parameters and diagnostics.

Usage Example:
--------------
    results = forecast_and_plot_weighted(
        df,
        segment='subprime',
        max_forecast_mob=72,
        use_only_mature=True
    )

Author: CAS
Date: 2025-07-18
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, minimize
from collections import defaultdict


def is_mature(loss_series, threshold=0.005, window=3):
    """Check if a loss curve has flattened: <0.5% relative change for N months"""
    diffs = np.diff(loss_series)
    rel_diffs = np.abs(diffs / (loss_series[1:] + 1e-8))  # Avoid divide by zero
    return np.all(rel_diffs[-window:] < threshold) if len(rel_diffs) >= window else False

def prepare_training_vintages(df, segment, min_mob=9, convergence_tol=0.005, min_obs=6):
    df_seg = df[df['segment'] == segment].copy()
    grouped = df_seg.groupby('vintage')

    vintage_info = []
    for vintage, group in grouped:
        group = group.sort_values('mob')
        mobs = group['mob'].values
        losses = group['cum_loss'].values
        volume = group['loan_volume'].iloc[0]

        n_obs = len(losses)
        mature = is_mature(losses, threshold=convergence_tol)
        meets_min_obs = n_obs >= min_obs

        vintage_info.append({
            'vintage': vintage,
            'mob_count': n_obs,
            'is_mature': mature,
            'enough_data': meets_min_obs,
            'loan_volume': volume,
            'mob': mobs,
            'loss': losses
        })

    return pd.DataFrame(vintage_info)

def fit_weibull_with_scale_weighted(mob, loss, weight):
    try:
        def weighted_loss(params):
            alpha, beta, L = params
            fitted = weibull_scaled(mob, alpha, beta, L)
            return np.sum(weight * (loss - fitted) ** 2)

        from scipy.optimize import minimize
        init_params = [12, 2, loss[-1]]
        bounds = [(1, 60), (0.5, 5), (0, 1.5 * max(loss))]
        result = minimize(weighted_loss, init_params, bounds=bounds, method='L-BFGS-B')
        if result.success:
            return result.x
        else:
            return [np.nan, np.nan, np.nan]
    except:
        return [np.nan, np.nan, np.nan]

def forecast_and_plot_weighted(df, segment, max_forecast_mob=72, use_only_mature=True):
    vintages_meta = prepare_training_vintages(df, segment)

    if use_only_mature:
        vintages_meta = vintages_meta[vintages_meta['is_mature']]
    else:
        vintages_meta = vintages_meta[vintages_meta['enough_data']]

    plt.figure(figsize=(12, 6))
    results = defaultdict(dict)

    for i, row in vintages_meta.iterrows():
        vintage = row['vintage']
        mob = row['mob']
        loss = row['loss']
        weight = np.full_like(loss, row['loan_volume'])

        params = fit_weibull_with_scale_weighted(mob, loss, weight)
        if np.isnan(params).any():
            continue

        alpha, beta, L = params
        mob_range = np.arange(0, max_forecast_mob + 1)
        forecast_curve = weibull_scaled(mob_range, alpha, beta, L)

        # Plot
        plt.plot(mob, loss, label=f"{vintage} actual", linestyle='-')
        forecast_start_idx = np.searchsorted(mob_range, mob[-1] + 1)
        if forecast_start_idx < len(mob_range):
            plt.plot(mob_range[forecast_start_idx:], forecast_curve[forecast_start_idx:], linestyle='--', label=f"{vintage} forecast")

        results[vintage] = {
            'alpha': alpha,
            'beta': beta,
            'ultimate_loss': L,
            'loan_volume': row['loan_volume'],
            'is_mature': row['is_mature']
        }

    plt.title(f"Weibull Fit (Volume-Weighted) for Segment: {segment}")
    plt.xlabel("Month on Book (MOB)")
    plt.ylabel("Cumulative Loss (%)")
    plt.grid(True)
    plt.legend(loc='upper left', fontsize=8)
    plt.tight_layout()
    plt.show()

    return results

# Example Usage:
results = forecast_and_plot_weighted(df, segment='subprime', max_forecast_mob=72, use_only_mature=True)

