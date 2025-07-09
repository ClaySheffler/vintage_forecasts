import pandas as pd
import numpy as np
from typing import Optional

class WideForecaster:
    """
    Forecasts cumulative gross charge-off rates using wide-format (one row per loan) data.
    Assumes columns: LOAN_ID, VINTAGE_DATE, LOAN_AMOUNT, MAX_REPORT_DATE, CHARGE_OFF_DATE (optional), CHARGE_OFF_AMOUNT (optional)
    """
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.df['VINTAGE_DATE'] = pd.to_datetime(self.df['VINTAGE_DATE'])
        self.df['MAX_REPORT_DATE'] = pd.to_datetime(self.df['MAX_REPORT_DATE'])
        if 'CHARGE_OFF_DATE' in self.df.columns:
            self.df['CHARGE_OFF_DATE'] = pd.to_datetime(self.df['CHARGE_OFF_DATE'])

    def cumulative_chargeoff_curve(self, groupby_col: Optional[str] = None, horizon_months: int = 60) -> pd.DataFrame:
        """
        Compute cumulative gross charge-off % curve by vintage or segment.
        Returns a DataFrame with columns: [GROUP, MONTH, CGCO_PCT]
        """
        results = []
        if groupby_col is None:
            group_keys = [('', self.df)]
        else:
            group_keys = self.df.groupby(groupby_col)
        for group, group_df in group_keys:
            vintage_date = group_df['VINTAGE_DATE'].min()
            for m in range(1, horizon_months+1):
                cutoff = vintage_date + pd.DateOffset(months=m)
                # Loans charged off by this month
                if 'CHARGE_OFF_DATE' in group_df.columns:
                    charged_off = group_df[(~group_df['CHARGE_OFF_DATE'].isna()) & (group_df['CHARGE_OFF_DATE'] <= cutoff)]['CHARGE_OFF_AMOUNT'].sum()
                else:
                    charged_off = 0
                orig_amt = group_df['LOAN_AMOUNT'].sum()
                cgco_pct = charged_off / orig_amt if orig_amt > 0 else np.nan
                results.append({
                    groupby_col or 'ALL': group,
                    'MONTH': m,
                    'CGCO_PCT': cgco_pct
                })
        return pd.DataFrame(results)

    def forecast_final_cgco(self, groupby_col: Optional[str] = None) -> pd.DataFrame:
        """
        Compute final cumulative gross charge-off % by group (e.g., vintage, FICO band).
        """
        if groupby_col is None:
            orig_amt = self.df['LOAN_AMOUNT'].sum()
            charged_off = self.df['CHARGE_OFF_AMOUNT'].sum()
            cgco_pct = charged_off / orig_amt if orig_amt > 0 else np.nan
            return pd.DataFrame([{'CGCO_PCT': cgco_pct}])
        else:
            grouped = self.df.groupby(groupby_col).agg({
                'LOAN_AMOUNT': 'sum',
                'CHARGE_OFF_AMOUNT': 'sum'
            }).reset_index()
            grouped['CGCO_PCT'] = grouped['CHARGE_OFF_AMOUNT'] / grouped['LOAN_AMOUNT']
            return grouped[[groupby_col, 'CGCO_PCT']]

# Example usage:
# loader = WideLoanDataLoader('loans_wide.csv')
# df = loader.get_dataframe()
# forecaster = WideForecaster(df)
# curve = forecaster.cumulative_chargeoff_curve(groupby_col='VINTAGE_DATE') 