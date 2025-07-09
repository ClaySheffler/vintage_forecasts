# %% [markdown]
# # Standalone Vintage Charge-off Forecasting Demo (Wide Format)
#
# This notebook demonstrates the workflow for cumulative gross charge-off forecasting using the new wide (one row per loan) format.

# %% [markdown]
# ## 1. Imports and Setup

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.wide_format.data_loader_wide import WideLoanDataLoader
from src.wide_format.forecaster_wide import WideForecaster
from src.shared.utils import assign_fico_band

# %% [markdown]
# ## 2. Create Synthetic Wide Format Data (for Demo)

# %%
# Create a small synthetic dataset
np.random.seed(42)
N = 500
vintages = pd.date_range('2020-01-01', periods=5, freq='QS')
loans = []
for i in range(N):
    vintage = np.random.choice(vintages)
    loan_amt = np.random.randint(5000, 25000)
    fico = np.random.randint(600, 820)
    max_report = vintage + pd.DateOffset(months=60)
    # 20% charge-off rate, random timing
    if np.random.rand() < 0.2:
        co_month = np.random.randint(6, 60)
        co_date = vintage + pd.DateOffset(months=co_month)
        co_amt = loan_amt * np.random.uniform(0.5, 1.0)
    else:
        co_date = pd.NaT
        co_amt = 0
    loans.append({
        'LOAN_ID': f'L{i:05d}',
        'VINTAGE_DATE': vintage,
        'LOAN_AMOUNT': loan_amt,
        'MAX_REPORT_DATE': max_report,
        'CHARGE_OFF_DATE': co_date,
        'CHARGE_OFF_AMOUNT': co_amt,
        'FICO_SCORE': fico
    })
df_wide = pd.DataFrame(loans)
df_wide['FICO_BAND'] = df_wide['FICO_SCORE'].apply(assign_fico_band)

# %% [markdown]
# ## 3. Load Data Using WideLoanDataLoader

# %%
# For demo, use the DataFrame directly (normally, save/load from CSV)
df_wide.to_csv('synthetic_loans_wide.csv', index=False)
loader = WideLoanDataLoader('synthetic_loans_wide.csv')
df = loader.get_dataframe()
display(df.head())

# %% [markdown]
# ## 4. Forecast Cumulative Gross Charge-off Curves

# %%
forecaster = WideForecaster(df)
curve = forecaster.cumulative_chargeoff_curve(groupby_col='VINTAGE_DATE', horizon_months=60)

# %% [markdown]
# ## 5. Plot Cumulative Charge-off Curves by Vintage

# %%
plt.figure(figsize=(10, 6))
for vintage, group in curve.groupby('VINTAGE_DATE'):
    plt.plot(group['MONTH'], group['CGCO_PCT'], label=str(vintage.date()))
plt.title('Cumulative Gross Charge-off % by Vintage (Wide Format)')
plt.xlabel('Seasoning Month')
plt.ylabel('Cumulative Gross CO%')
plt.legend(title='Vintage')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 6. Final CGCO% by Vintage

# %%
final_cgco = forecaster.forecast_final_cgco(groupby_col='VINTAGE_DATE')
display(final_cgco)

# %% [markdown]
# ## 7. Summary
#
# - This workflow uses the new wide format for efficient, scalable forecasting.
# - For advanced time series or delinquency analysis, use the preserved long format modules in `src/long_format/`. 