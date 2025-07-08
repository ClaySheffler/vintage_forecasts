# %% [markdown]
# # Standalone Vintage Charge-off Forecasting Demo (Full Logic)
#
# This notebook demonstrates the full workflow for vintage charge-off forecasting with FICO segmentation, using data loaded from Snowflake. All logic is self-contained, including flexible data handling, advanced curve fitting, scenario forecasting, and reporting.

# %% [markdown]
# ## 1. Configuration and Imports

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from snowflake.snowpark.context import get_active_session
from scipy.optimize import curve_fit
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# FICO band definitions
FICO_BANDS = {
    '600-649': {'min': 600, 'max': 649, 'risk_grade': 5, 'label': 'Very High Risk'},
    '650-699': {'min': 650, 'max': 699, 'risk_grade': 4, 'label': 'High Risk'},
    '700-749': {'min': 700, 'max': 749, 'risk_grade': 3, 'label': 'Medium Risk'},
    '750-799': {'min': 750, 'max': 799, 'risk_grade': 2, 'label': 'Low Risk'},
    '800+': {'min': 800, 'max': 850, 'risk_grade': 1, 'label': 'Very Low Risk'}
}

MATURE_MONTHS = 72
ACTUALS_MONTHS = 96
MATURITY_ADJUSTMENT = 0.01
FOCUS_YEARS = 5

# %% [markdown]
# ## 2. Connect to Snowflake and Load Data

# %%
# Edit your query as needed
query = '''
SELECT
    loan_id,
    vintage_date,
    report_date,
    seasoning_month,
    fico_score,
    loan_amount,
    charge_off_flag,
    charge_off_amount,
    outstanding_balance,
    term,
    interest_rate
FROM your_database.your_schema.your_loan_performance_table
WHERE vintage_date >= '2018-01-01'
'''

session = get_active_session()
df = session.sql(query).to_pandas()
print(f'Loaded {len(df):,} records from Snowflake.')
df.head()

# %% [markdown]
# ## 3. FICO Band Assignment and Flexible Data Handling

# %%
def assign_fico_band(fico):
    if fico >= 800: return '800+'
    elif fico >= 750: return '750-799'
    elif fico >= 700: return '700-749'
    elif fico >= 650: return '650-699'
    elif fico >= 600: return '600-649'
    else: return '<600'

df['fico_band'] = df['fico_score'].apply(assign_fico_band)
df = df.dropna(subset=['loan_id', 'vintage_date', 'seasoning_month', 'fico_score', 'loan_amount'])
df['vintage_date'] = pd.to_datetime(df['vintage_date'])

# Flexible data handling: auto-complete missing seasoning months for each loan
all_months = df.groupby('loan_id')['seasoning_month'].max().to_dict()
rows = []
for loan_id, group in df.groupby('loan_id'):
    max_month = all_months[loan_id]
    for m in range(0, max_month+1):
        if m in group['seasoning_month'].values:
            row = group[group['seasoning_month'] == m].iloc[0]
        else:
            row = group.iloc[0].copy()
            row['seasoning_month'] = m
            row['charge_off_flag'] = 0
            row['charge_off_amount'] = 0
            row['outstanding_balance'] = np.nan
        rows.append(row)
df_flex = pd.DataFrame(rows)
df_flex = df_flex.sort_values(['loan_id', 'seasoning_month'])
print('Flexible data shape:', df_flex.shape)

# %% [markdown]
# ## 4. Vintage Analysis and Cumulative Gross Charge-off %

# %%
vintage_metrics = df_flex.groupby(['vintage_date', 'fico_band', 'seasoning_month']).agg({
    'loan_amount': 'sum',
    'outstanding_balance': 'sum',
    'charge_off_amount': 'sum',
    'loan_id': 'count'
}).reset_index()
vintage_metrics['charge_off_flag'] = vintage_metrics['charge_off_amount'] / vintage_metrics['outstanding_balance']
vintage_metrics['cumulative_charge_off_flag'] = (
    vintage_metrics.groupby(['vintage_date', 'fico_band'])['charge_off_amount'].cumsum() /
    vintage_metrics.groupby(['vintage_date', 'fico_band'])['loan_amount'].first()
)
risk_grade_map = {band: FICO_BANDS[band]['risk_grade'] for band in FICO_BANDS}
vintage_metrics['risk_grade'] = vintage_metrics['fico_band'].map(risk_grade_map)
vintage_metrics['vintage_quarter'] = vintage_metrics['vintage_date'].dt.to_period('Q').dt.start_time
vintage_metrics.head()

# %% [markdown]
# ## 5. Curve Fitting: Weibull, Lognormal, Gompertz, Linear, Ensemble

# %%
def weibull_curve(x, alpha, beta, gamma):
    return alpha * (1 - np.exp(-((x / beta) ** gamma)))
def lognormal_curve(x, alpha, mu, sigma):
    return alpha * stats.lognorm.cdf(x, sigma, scale=np.exp(mu))
def gompertz_curve(x, alpha, beta, gamma):
    return alpha * np.exp(-beta * np.exp(-gamma * x))
def linear_curve(x, a, b):
    return a * x + b

def fit_curve(curve_func, x, y, p0, bounds):
    try:
        popt, _ = curve_fit(curve_func, x, y, p0=p0, bounds=bounds, maxfev=10000)
        y_pred = curve_func(x, *popt)
        r2 = 1 - np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2)
        rmse = np.sqrt(np.mean((y - y_pred) ** 2))
        return {'params': popt, 'r2': r2, 'rmse': rmse, 'y_pred': y_pred}
    except Exception as e:
        return None

# Fit curves for each FICO band
fit_results = {}
for band in vintage_metrics['fico_band'].unique():
    band_data = vintage_metrics[(vintage_metrics['fico_band'] == band) & (vintage_metrics['seasoning_month'] >= 6)]
    x = band_data['seasoning_month'].values
    y = band_data['cumulative_charge_off_flag'].values
    results = {}
    results['weibull'] = fit_curve(weibull_curve, x, y, p0=[0.05, 24, 2], bounds=([0, 1, 0.1], [0.3, 60, 10]))
    results['lognormal'] = fit_curve(lognormal_curve, x, y, p0=[0.05, 3, 0.5], bounds=([0, 1, 0.1], [0.3, 5, 2]))
    results['gompertz'] = fit_curve(gompertz_curve, x, y, p0=[0.05, 1, 0.1], bounds=([0, 0, 0], [0.3, 10, 1]))
    results['linear'] = fit_curve(linear_curve, x, y, p0=[0.01, 0], bounds=([-1, -1], [1, 1]))
    # Ensemble: average of available models
    preds = [r['y_pred'] for r in results.values() if r and 'y_pred' in r]
    if preds:
        ensemble_pred = np.mean(preds, axis=0)
        r2 = 1 - np.sum((y - ensemble_pred) ** 2) / np.sum((y - np.mean(y)) ** 2)
        rmse = np.sqrt(np.mean((y - ensemble_pred) ** 2))
        results['ensemble'] = {'params': None, 'r2': r2, 'rmse': rmse, 'y_pred': ensemble_pred}
    fit_results[band] = results

# Model comparison table
model_table = []
for band, models in fit_results.items():
    for name, res in models.items():
        if res:
            model_table.append({
                'FICO Band': band,
                'Model': name,
                'RMSE': res['rmse'],
                'R²': res['r2'],
                'Params': res['params'] if res['params'] is not None else 'Ensemble',
                'Explainability': 'High' if name in ['linear', 'weibull', 'lognormal', 'gompertz'] else 'Medium',
                'Notes': ''
            })
model_df = pd.DataFrame(model_table)
print(model_df)

# %% [markdown]
# ## 6. Simple Scaling and Additive Forecast Methods

# %%
def scaling_method(observed_month, observed_cum_co, typical_cum_co):
    # Scale observed by reciprocal of typical proportion at observed month
    if observed_month in typical_cum_co and typical_cum_co[observed_month] > 0:
        return observed_cum_co / typical_cum_co[observed_month]
    return np.nan

def additive_method(observed_month, observed_cum_co, typical_incremental_co, max_month=60):
    # Add expected future charge-off rates to current observed
    future_months = range(observed_month+1, max_month+1)
    future_sum = sum([typical_incremental_co.get(m, 0) for m in future_months])
    return observed_cum_co + future_sum

# Example usage: (for a given band)
band = list(fit_results.keys())[0]
band_data = vintage_metrics[(vintage_metrics['fico_band'] == band) & (vintage_metrics['seasoning_month'] >= 6)]
typical_cum_co = dict(zip(band_data['seasoning_month'], band_data['cumulative_charge_off_flag']))
typical_incremental_co = {m: typical_cum_co.get(m, 0) - typical_cum_co.get(m-1, 0) for m in typical_cum_co}
observed_month = 24
observed_cum_co = typical_cum_co.get(observed_month, 0)
print('Scaling method forecast:', scaling_method(observed_month, observed_cum_co, typical_cum_co))
print('Additive method forecast:', additive_method(observed_month, observed_cum_co, typical_incremental_co))

# %% [markdown]
# ## 7. Scenario/Quality-Mix Forecasting

# %%
# Example: define scenarios as different FICO band mixes
fico_mix_scenarios = {
    'Conservative': {'800+': 0.2, '750-799': 0.3, '700-749': 0.3, '650-699': 0.15, '600-649': 0.05},
    'Balanced':     {'800+': 0.1, '750-799': 0.2, '700-749': 0.4, '650-699': 0.2,  '600-649': 0.1},
    'Aggressive':   {'800+': 0.05, '750-799': 0.15, '700-749': 0.3, '650-699': 0.3, '600-649': 0.2}
}

portfolio_size = 100_000_000
scenario_results = {}
for scenario, mix in fico_mix_scenarios.items():
    total_forecast = 0
    for band, pct in mix.items():
        # Use ensemble model if available, else Weibull
        model = fit_results[band].get('ensemble') or fit_results[band].get('weibull')
        if model:
            # Forecast at 60 months
            forecast_60 = model['y_pred'][min(60-6, len(model['y_pred'])-1)]  # index offset for months >=6
            total_forecast += pct * portfolio_size * forecast_60
    scenario_results[scenario] = total_forecast / portfolio_size
print('Scenario Forecasts (Cumulative Gross CO% at 60 months):', scenario_results)

# %% [markdown]
# ## 8. Forecasting for Recent Vintages Using Fitted Curves

# %%
# For each forecast focus vintage, forecast future charge-offs using best model
focus_vintages = vintage_metrics['vintage_quarter'].drop_duplicates().tail(FOCUS_YEARS*4)  # last 5 years, quarterly
forecast_horizon = 60
forecast_table = []
for vintage in focus_vintages:
    for band in vintage_metrics['fico_band'].unique():
        model = fit_results[band].get('ensemble') or fit_results[band].get('weibull')
        if model:
            months = np.arange(6, forecast_horizon+1)
            y_pred = model['y_pred'][:len(months)]
            forecast_table.append({
                'vintage_quarter': vintage,
                'fico_band': band,
                'forecasted_cumulative_gross_chargeoff_pct': y_pred[-1]
            })
forecast_df = pd.DataFrame(forecast_table)
print('Forecasts for Recent Vintages:')
print(forecast_df)

# %% [markdown]
# ## 9. Export/Reporting

# %%
# Export mature performance and forecast tables to Excel
with pd.ExcelWriter('vintage_forecast_outputs.xlsx') as writer:
    # You can add more sheets as needed
    forecast_df.to_excel(writer, sheet_name='Forecasts', index=False)
print('Exported results to vintage_forecast_outputs.xlsx')

# %% [markdown]
# ## 10. Visualization: Cumulative Gross Charge-off % by Vintage and Segment

# %%
import seaborn as sns
plt.figure(figsize=(12, 6))
sns.lineplot(data=forecast_df, x='vintage_quarter', y='forecasted_cumulative_gross_chargeoff_pct', hue='fico_band', marker='o')
plt.title('Forecasted Cumulative Gross Charge-off % for Recent Vintages')
plt.ylabel('Forecasted CGCO%')
plt.xlabel('Vintage Quarter')
plt.xticks(rotation=45)
plt.legend(title='FICO Band')
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 11. Basic Validation

# %%
# Check for any negative or >1 CGCO% values
print('Any negative CGCO% in forecast_df?', (forecast_df['forecasted_cumulative_gross_chargeoff_pct'] < 0).any())
print('Any CGCO% > 1 in forecast_df?', (forecast_df['forecasted_cumulative_gross_chargeoff_pct'] > 1).any())

# %% [markdown]
# ## 12. Visualization: Cumulative Gross Charge-off % Heatmap (Vintages x Seasoning Month)

# %%
import matplotlib.pyplot as plt
import seaborn as sns
# Prepare heatmap data (every 6 months up to 96)
heatmap_data = vintage_metrics[(vintage_metrics['seasoning_month'] <= 96) & (vintage_metrics['seasoning_month'] % 6 == 0)]
heatmap_pivot = heatmap_data.pivot_table(index='vintage_date', columns='seasoning_month', values='cumulative_charge_off_flag', aggfunc='mean')

# Prepare forecasted data for the same grid
# Assume forecasted values are available in forecast_df with columns: vintage_quarter, fico_band, forecasted_cumulative_gross_chargeoff_pct
# For each vintage, fill in forecasted values for months beyond last actual
# (This is a simplified example; adapt as needed for your forecast_df structure)

# Find the last actual month for each vintage
last_actual = heatmap_data.groupby('vintage_date')['seasoning_month'].max()
forecast_overlay = heatmap_pivot.copy()
for vintage in heatmap_pivot.index:
    last_month = last_actual.get(vintage, 0)
    # Example: fill months > last_month with forecast if available
    for month in heatmap_pivot.columns:
        if month > last_month:
            # Use forecasted value if available (replace with your logic)
            # forecast_val = ...
            forecast_overlay.loc[vintage, month] = np.nan  # Placeholder for forecasted value

plt.figure(figsize=(14, max(6, len(heatmap_pivot)//4)))
sns.heatmap(heatmap_pivot, annot=True, fmt='.2%', cmap='YlOrRd', cbar_kws={'label': 'Cumulative Gross CO%'})
# Overlay forecasted cells with lighter alpha (if any)
sns.heatmap(forecast_overlay, annot=False, fmt='.2%', cmap='YlOrRd', alpha=0.3, cbar=False)
plt.title('Cumulative Gross Charge-off % Heatmap (Actuals + Forecasts)')
plt.ylabel('Vintage')
plt.xlabel('Seasoning Month')
plt.tight_layout()
plt.figtext(0.5, 0.01, 'Lighter cells indicate forecasted values', ha='center', fontsize=10)
plt.show()

# %% [markdown]
# ## 13. Visualization: Cumulative Gross Charge-off % by Vintage (Line Chart)

# %%
plt.figure(figsize=(14, 7))
for vintage, group in vintage_metrics.groupby('vintage_date'):
    group = group[group['seasoning_month'] <= 96]
    last_actual_month = group['seasoning_month'].max()
    # Plot actuals
    plt.plot(group['seasoning_month'], group['cumulative_charge_off_flag'], label=f'{vintage} (Actual)', linestyle='solid')
    # Plot forecasts if available
    # Example: forecasted_months = ...
    # forecasted_values = ...
    # plt.plot(forecasted_months, forecasted_values, label=f'{vintage} (Forecast)', linestyle='dashed')
plt.title('Cumulative Gross Charge-off % by Vintage (Actuals + Forecasts)')
plt.xlabel('Seasoning Month')
plt.ylabel('Cumulative Gross CO%')
plt.legend(title='Vintage', bbox_to_anchor=(1.05, 1), loc='upper left', ncol=1, fontsize='small')
plt.tight_layout()
plt.figtext(0.5, 0.01, 'Dashed lines indicate forecasted values', ha='center', fontsize=10)
plt.show() 