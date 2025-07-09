# Long Format Guide

## Overview

The long (panel) format is a one-row-per-loan-per-period data structure. It is required for advanced time series analysis, delinquency/prepayment modeling, and full balance tracking.

## When to Use
- You need to analyze loan performance over time (not just final outcomes)
- You want to model delinquency, prepayment, or roll rates
- You need to track actual outstanding balances, payment history, or dynamic loan states
- Regulatory or advanced risk modeling requirements

## What the Long Format Enables
- True vintage/seasoning curve analysis (actuals, not just cumulative)
- Roll rate and transition analysis
- Prepayment and early payoff modeling
- Delinquency and recovery analysis
- Dynamic cohort and state tracking

## Example Data

| LOAN_ID | VINTAGE_DATE | REPORT_DATE | SEASONING_MONTH | FICO_SCORE | LOAN_AMOUNT | OUTSTANDING_BALANCE | CHARGE_OFF_FLAG | ... |
|---------|--------------|-------------|-----------------|------------|-------------|--------------------|-----------------|-----|
| L00001  | 2020-01-01   | 2020-01-31  | 0               | 710        | 10000       | 10000              | 0               | ... |
| L00001  | 2020-01-01   | 2020-02-29  | 1               | 710        | 10000       | 9950               | 0               | ... |
| L00001  | 2020-01-01   | 2022-06-30  | 29              | 710        | 10000       | 0                  | 1               | ... |

## How to Use

- Use the modules in `src/long_format/` for loading, validating, and analyzing long format data.
- For legacy/flexible data handling, use the preserved code from the original project.
- For new projects, prefer the wide format unless you need full time series detail.

## Migration Notes
- To convert from wide to long format, you must reconstruct the time series (using amortization schedules or external data).
- For large datasets, the long format can be slow and memory-intensive. Use with caution for portfolios >1M loans.

## Performance Warning
- The long format is much larger and slower than the wide format.
- Only use when you need full time series detail.

## For Efficient Forecasting
- Use the wide format modules in `src/wide_format/` for cumulative loss forecasting and reporting. 