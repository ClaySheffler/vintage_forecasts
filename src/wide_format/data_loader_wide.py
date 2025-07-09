import pandas as pd
from typing import Optional, List

class WideLoanDataLoader:
    """
    Loads and validates wide-format (one row per loan) loan data for vintage forecasting.
    Required columns:
        - LOAN_ID
        - VINTAGE_DATE
        - LOAN_AMOUNT
        - MAX_REPORT_DATE
        - (Optional) CHARGE_OFF_DATE
        - (Optional) CHARGE_OFF_AMOUNT
        - (Optional) FICO_SCORE, TERM, etc.
    """
    REQUIRED_COLUMNS = ['LOAN_ID', 'VINTAGE_DATE', 'LOAN_AMOUNT', 'MAX_REPORT_DATE']

    def __init__(self, file_path: str, file_type: str = 'csv', extra_columns: Optional[List[str]] = None):
        self.file_path = file_path
        self.file_type = file_type
        self.extra_columns = extra_columns or []
        self.data = self._load_data()

    def _load_data(self) -> pd.DataFrame:
        if self.file_type == 'csv':
            df = pd.read_csv(self.file_path)
        elif self.file_type == 'parquet':
            df = pd.read_parquet(self.file_path)
        elif self.file_type == 'excel':
            df = pd.read_excel(self.file_path)
        else:
            raise ValueError(f"Unsupported file type: {self.file_type}")
        df.columns = [col.upper() for col in df.columns]
        self._validate_columns(df)
        return df

    def _validate_columns(self, df: pd.DataFrame):
        missing = [col for col in self.REQUIRED_COLUMNS if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        # Check for duplicate LOAN_IDs
        if df['LOAN_ID'].duplicated().any():
            raise ValueError("Duplicate LOAN_IDs found in input data.")

    def get_dataframe(self) -> pd.DataFrame:
        """Return the validated wide-format DataFrame."""
        return self.data

    def get_loan_ids(self) -> List[str]:
        return self.data['LOAN_ID'].tolist()

    def get_column(self, col: str) -> pd.Series:
        return self.data[col.upper()]

# Example usage:
# loader = WideLoanDataLoader('loans_wide.csv')
# df = loader.get_dataframe() 