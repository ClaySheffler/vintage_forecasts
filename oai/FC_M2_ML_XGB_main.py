"""
Loss Curve Forecasting via Curve Decomposition + Machine Learning (Method #2)
-------------------------------------------------------------------------------

This script implements an end-to-end modeling pipeline using a two-step approach:
1. Decompose each vintage's loss curve into:
   - A normalized shape curve (percentage of ultimate loss)
   - An estimated ultimate loss value (scale)
2. Use a machine learning model to predict the terminal loss (scale) using known vintage features.

This method allows flexible forecasting of incomplete loss curves, especially when loss shape is stable
and scale varies by credit quality, macro, or early performance.

Expected Input:
---------------
A pandas DataFrame `df` with the following columns:
- 'vintage': str or int, identifying the vintage (e.g., '2019Q2')
- 'mob': int, month-on-book
- 'cum_loss': float, cumulative loss percentage or ratio
- Additional columns for ML features (e.g., early MOB loss, origination FICO, DTI, macro indicators, etc.)

This script assumes the DataFrame is loaded and named `df`.

Dependencies:
-------------
- pandas, numpy, matplotlib, sklearn, xgboost

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# --- Step 1: Decompose loss curves ---
def prepare_curve_decomposition(df, min_mob=12):
    grouped = df.groupby('vintage')
    shape_matrix = {}
    terminal_losses = {}

    for vintage, group in grouped:
        mob_series = group.sort_values('mob')
        losses = mob_series['cum_loss'].values
        if len(losses) >= min_mob:
            terminal = losses[-1]
            shape = losses / terminal if terminal > 0 else np.zeros_like(losses)
            shape_matrix[vintage] = shape
            terminal_losses[vintage] = terminal

    # Build aligned shape DataFrame
    max_mob = max(len(v) for v in shape_matrix.values())
    shape_df = pd.DataFrame({k: np.pad(v, (0, max_mob - len(v)), constant_values=np.nan)
                             for k, v in shape_matrix.items()}).T
    shape_df.columns = [f"mob_{i}" for i in range(max_mob)]

    target = pd.Series(terminal_losses)
    return shape_df, target

# --- Step 2: Train ML model to predict ultimate loss ---
def train_terminal_loss_model(df_features, target):
    X_train, X_test, y_train, y_test = train_test_split(df_features, target, test_size=0.2, random_state=42)
    model = XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"RMSE on holdout: {mean_squared_error(y_test, y_pred, squared=False):.4f}")
    return model

# --- Step 3: Forecast curves using model + historical shape ---
def forecast_loss_curve(df_new, shape_df, model):
    avg_shape = shape_df.mean(skipna=True).values
    mob_labels = shape_df.columns

    # Predict terminal losses
    preds = model.predict(df_new)
    curves = []

    for i, L in enumerate(preds):
        curve = L * avg_shape
        curves.append(curve)

    df_curve = pd.DataFrame(curves, columns=mob_labels)
    df_curve['vintage'] = df_new.index
    return df_curve.set_index('vintage')

# --- Step 4: Plot actual vs forecasted loss curves ---
def plot_curve_forecasts(df_actual, df_forecast):
    plt.figure(figsize=(12, 6))
    for vintage in df_forecast.index:
        if vintage in df_actual['vintage'].unique():
            actual = df_actual[df_actual['vintage'] == vintage].sort_values('mob')
            forecast = df_forecast.loc[vintage].values
            mob_range = np.arange(len(forecast))

            known_mob = actual['mob'].values
            known_loss = actual['cum_loss'].values
            plt.plot(known_mob, known_loss, linestyle='-', label=f"{vintage} actual")
            plt.plot(mob_range[len(known_mob):], forecast[len(known_mob):], linestyle='--', label=f"{vintage} forecast")

    plt.title("Actual vs Forecasted Loss Curves (ML + Shape Model)")
    plt.xlabel("Month on Book (MOB)")
    plt.ylabel("Cumulative Loss")
    plt.grid(True)
    plt.legend(fontsize=8, loc='upper left')
    plt.tight_layout()
    plt.show()

# --- Main Execution Block ---
if __name__ == "__main__":
    # Load or assume df is defined in scope
    # df = pd.read_csv("loss_data.csv")

    # Example: Engineer features (assumes loss data and early MOBs are available)
    feature_df = df[df['mob'] <= 3].groupby('vintage').agg({
        'cum_loss': ['max'],
        'loan_volume': 'first',
        'fico_avg': 'first',
        'dti_avg': 'first'
    })
    fe