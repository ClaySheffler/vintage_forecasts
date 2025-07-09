# Wide Format Guide

## Overview

The wide format is a one-row-per-loan data structure designed for efficient, scalable cumulative gross charge-off forecasting. It is ideal for large datasets and high-performance workflows.

## Required Columns

- `LOAN_ID`: Unique loan identifier
- `VINTAGE_DATE`: Origination date
- `LOAN_AMOUNT`: Original loan amount
- `MAX_REPORT_DATE`: Last observed date for the loan
- (Optional) `CHARGE_OFF_DATE`: Date of charge-off (if applicable)
- (Optional) `CHARGE_OFF_AMOUNT`: Amount charged off (if applicable)
- (Optional) `FICO_SCORE`, `FICO_BAND`, `TERM`, etc.

## What You Can Do

- Cumulative gross charge-off forecasting
- Vintage/seasoning curve analysis (cumulative, not dynamic)
- Portfolio-level loss rates and segmentation
- Time-to-default/survival analysis

## What You Cannot Do

- Track actual outstanding balance over time (unless you assume a schedule)
- Analyze delinquencies, prepayments, or dynamic loan states
- Roll rate or transition analysis

## Example Data

| LOAN_ID | VINTAGE_DATE | LOAN_AMOUNT | MAX_REPORT_DATE | CHARGE_OFF_DATE | CHARGE_OFF_AMOUNT | FICO_SCORE |
|---------|--------------|-------------|-----------------|-----------------|-------------------|------------|
| L00001  | 2020-01-01   | 10000       | 2025-01-01      | 2022-06-01      | 8000              | 710        |
| L00002  | 2020-04-01   | 15000       | 2025-04-01      |                 | 0                 | 680        |

## How to Use

### 1. Load Data
```python
from src.wide_format.data_loader_wide import WideLoanDataLoader
loader = WideLoanDataLoader('loans_wide.csv')
df = loader.get_dataframe()
```

### 2. Forecast Cumulative Charge-off Curves
```python
from src.wide_format.forecaster_wide import WideForecaster
forecaster = WideForecaster(df)
curve = forecaster.cumulative_chargeoff_curve(groupby_col='VINTAGE_DATE', horizon_months=60)
```

### 3. Plot Results
```python
import matplotlib.pyplot as plt
for vintage, group in curve.groupby('VINTAGE_DATE'):
    plt.plot(group['MONTH'], group['CGCO_PCT'], label=str(vintage))
plt.legend()
plt.show()
```

## Migration Tips

- To convert from long to wide format, aggregate each loan's history to a single row, keeping origination, final status, and charge-off info.
- Use the wide format for all new forecasting and reporting workflows.
- Preserve the long format for advanced time series or delinquency analysis (see `long_format/`).

## Performance Benefits

- Much smaller in-memory data (1 row per loan)
- No more flexible data handling bottleneck
- Fast forecasting and reporting

## For Advanced Use Cases

- Use the preserved long format modules in `src/long_format/` for full balance detail, delinquency, or prepayment analysis. 