# %% [markdown]
# # Standalone Vintage Charge-off Forecasting Demo
#
# This notebook demonstrates the full workflow for vintage charge-off forecasting with FICO segmentation, using data loaded from Snowflake. All logic is self-contained.

# %% [markdown]
# ## 1. Configuration and Imports

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from snowflake.snowpark.context import get_active_session
from scipy.optimize import curve_fit
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
    seasoning_month,
    fico_score,
    loan_amount,
    charge_off_flag,
    charge_off_amount,
    outstanding_balance,
    term
FROM your_database.your_schema.your_loan_performance_table
WHERE vintage_date >= '2018-01-01'
'''

session = get_active_session()
df = session.sql(query).to_pandas()
print(f'Loaded {len(df):,} records from Snowflake.')
df.head()

# %% [markdown]
# ## 3. FICO Band Assignment and Preprocessing

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
print('FICO Band Distribution:')
print(df['fico_band'].value_counts())
df.head()

# %% [markdown]
# ## 4. Vintage Analysis and Cumulative Gross Charge-off %

# %%
# Calculate vintage-level metrics
vintage_metrics = df.groupby(['vintage_date', 'fico_band', 'seasoning_month']).agg({
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
# Add risk grade
risk_grade_map = {band: FICO_BANDS[band]['risk_grade'] for band in FICO_BANDS}
vintage_metrics['risk_grade'] = vintage_metrics['fico_band'].map(risk_grade_map)
vintage_metrics['vintage_quarter'] = vintage_metrics['vintage_date'].dt.to_period('Q').dt.start_time
vintage_metrics.head()

# %% [markdown]
# ## 5. Mature Vintage and Forecast Focus Logic

# %%
today = pd.Timestamp(datetime.today().date())

def get_mature_vintage_performance(vintage_metrics, mature_months=72, actuals_months=96, adjustment=0.01):
    results = []
    for (vintage, band), group in vintage_metrics.groupby(['vintage_quarter', 'fico_band']):
        max_seasoning = group['seasoning_month'].max()
        vintage_age_months = ((today - pd.to_datetime(vintage)).days // 30)
        if vintage_age_months >= actuals_months:
            final_cgco = group.loc[group['seasoning_month'] == max_seasoning, 'cumulative_charge_off_flag'].values[0]
        elif mature_months <= vintage_age_months < actuals_months:
            cgco_72 = group.loc[group['seasoning_month'] == mature_months, 'cumulative_charge_off_flag']
            if not cgco_72.empty:
                final_cgco = cgco_72.values[0] + adjustment
            else:
                final_cgco = None
        else:
            final_cgco = None
        results.append({'vintage_quarter': vintage, 'fico_band': band, 'final_cumulative_gross_chargeoff_pct': final_cgco})
    return pd.DataFrame(results)

def get_forecast_focus_vintages(vintage_metrics, focus_years=5):
    cutoff = today - pd.DateOffset(years=focus_years)
    focus_vintages = vintage_metrics['vintage_quarter'][pd.to_datetime(vintage_metrics['vintage_quarter']) >= cutoff].unique()
    return list(focus_vintages)

mature_perf = get_mature_vintage_performance(vintage_metrics)
focus_vintages = get_forecast_focus_vintages(vintage_metrics)
print('Mature Vintage Performance Table (Final CGCO%):')
print(mature_perf)
print('Forecast Focus Vintages (last 5 years, quarterly):')
print(focus_vintages)

# %% [markdown]
# ## 6. Visualization: Cumulative Gross Charge-off % by Vintage and Segment

# %%
import seaborn as sns
plt.figure(figsize=(12, 6))
sns.lineplot(data=mature_perf, x='vintage_quarter', y='final_cumulative_gross_chargeoff_pct', hue='fico_band', marker='o')
plt.title('Final Cumulative Gross Charge-off % by Vintage Quarter and FICO Band')
plt.ylabel('Final CGCO%')
plt.xlabel('Vintage Quarter')
plt.xticks(rotation=45)
plt.legend(title='FICO Band')
plt.tight_layout()
plt.show() 