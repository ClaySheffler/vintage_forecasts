"""
Data loader for vintage charge-off forecasting.
Handles loading and preprocessing of historical loan performance data.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


class LoanDataLoader:
    """
    Loads and preprocesses loan performance data for vintage analysis.
    """
    
    def __init__(self, data_path: Optional[str] = None):
        """
        Initialize the data loader.
        
        Args:
            data_path: Path to the loan performance data file
        """
        self.data_path = data_path
        self.data = None
        
    def load_sample_data(self) -> pd.DataFrame:
        """
        Generate sample loan performance data for demonstration.
        In production, this would load from actual data sources.
        
        Returns:
            DataFrame with loan performance data
        """
        # Generate sample data spanning from 2014 to present
        np.random.seed(42)  # For reproducible results
        
        # Create date range
        start_date = pd.Timestamp('2014-01-01')
        end_date = pd.Timestamp('2024-12-31')
        date_range = pd.date_range(start=start_date, end=end_date, freq='M')
        
        # Generate sample loan vintages (origination periods)
        vintage_periods = pd.date_range(start='2014-01-01', end='2023-12-31', freq='M')
        
        data_rows = []
        
        for vintage_date in vintage_periods:
            # Number of loans originated in this vintage
            num_loans = np.random.randint(1000, 5000)
            
            # Generate loan IDs for this vintage
            loan_ids = [f"LOAN_{vintage_date.strftime('%Y%m')}_{i:06d}" 
                       for i in range(num_loans)]
            
            # Loan characteristics
            loan_amounts = np.random.lognormal(mean=10.5, sigma=0.5, size=num_loans)
            interest_rates = np.random.normal(0.08, 0.02, num_loans)
            terms = np.random.choice([120, 180, 240, 300], size=num_loans, p=[0.3, 0.3, 0.2, 0.2])
            
            # Generate performance data for each loan over time
            for i, loan_id in enumerate(loan_ids):
                loan_amount = loan_amounts[i]
                interest_rate = interest_rates[i]
                term = terms[i]
                
                # Calculate seasoning months for this loan
                seasoning_months = []
                charge_off_rates = []
                outstanding_balances = []
                
                for report_date in date_range:
                    if report_date >= vintage_date:
                        seasoning_month = ((report_date.year - vintage_date.year) * 12 + 
                                         report_date.month - vintage_date.month)
                        
                        if seasoning_month <= term:
                            seasoning_months.append(seasoning_month)
                            
                            # Calculate outstanding balance
                            remaining_balance = loan_amount * (1 - seasoning_month / term)
                            outstanding_balances.append(remaining_balance)
                            
                            # Calculate charge-off rate based on vintage and seasoning
                            base_rate = 0.02  # 2% base charge-off rate
                            
                            # Vintage effect (some vintages perform better/worse)
                            vintage_effect = 0.5 + 0.5 * np.sin(vintage_date.month / 12 * 2 * np.pi)
                            
                            # Seasoning effect (charge-offs typically peak around 18-24 months)
                            seasoning_effect = np.exp(-((seasoning_month - 20) ** 2) / 100)
                            
                            # Economic cycle effect (simulate 2008-like crisis)
                            if 2018 <= vintage_date.year <= 2020:
                                crisis_effect = 1.5
                            else:
                                crisis_effect = 1.0
                            
                            # Random variation
                            random_effect = np.random.normal(1, 0.1)
                            
                            charge_off_rate = (base_rate * vintage_effect * seasoning_effect * 
                                             crisis_effect * random_effect)
                            charge_off_rate = max(0, min(0.15, charge_off_rate))  # Cap at 15%
                            
                            charge_off_rates.append(charge_off_rate)
                            
                            data_rows.append({
                                'loan_id': loan_id,
                                'vintage_date': vintage_date,
                                'report_date': report_date,
                                'seasoning_month': seasoning_month,
                                'loan_amount': loan_amount,
                                'interest_rate': interest_rate,
                                'term': term,
                                'outstanding_balance': remaining_balance,
                                'charge_off_rate': charge_off_rate,
                                'charge_off_amount': remaining_balance * charge_off_rate
                            })
        
        self.data = pd.DataFrame(data_rows)
        return self.data
    
    def load_from_file(self, file_path: str, file_type: str = 'csv') -> pd.DataFrame:
        """
        Load data from file.
        
        Args:
            file_path: Path to the data file
            file_type: Type of file ('csv', 'excel', 'parquet')
            
        Returns:
            DataFrame with loan performance data
        """
        if file_type.lower() == 'csv':
            self.data = pd.read_csv(file_path)
        elif file_type.lower() == 'excel':
            self.data = pd.read_excel(file_path)
        elif file_type.lower() == 'parquet':
            self.data = pd.read_parquet(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
        
        return self.data
    
    def preprocess_data(self) -> pd.DataFrame:
        """
        Preprocess the loaded data for vintage analysis.
        
        Returns:
            Preprocessed DataFrame
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_sample_data() or load_from_file() first.")
        
        # Convert date columns
        date_columns = ['vintage_date', 'report_date']
        for col in date_columns:
            if col in self.data.columns:
                self.data[col] = pd.to_datetime(self.data[col])
        
        # Sort by vintage_date and report_date
        self.data = self.data.sort_values(['vintage_date', 'report_date'])
        
        # Add additional features
        self.data['vintage_year'] = self.data['vintage_date'].dt.year
        self.data['vintage_month'] = self.data['vintage_date'].dt.month
        self.data['report_year'] = self.data['report_date'].dt.year
        self.data['report_month'] = self.data['report_date'].dt.month
        
        # Calculate vintage age in months
        self.data['vintage_age_months'] = (
            (self.data['report_date'].dt.year - self.data['vintage_date'].dt.year) * 12 +
            (self.data['report_date'].dt.month - self.data['vintage_date'].dt.month)
        )
        
        return self.data
    
    def get_vintage_summary(self) -> pd.DataFrame:
        """
        Create vintage summary statistics.
        
        Returns:
            DataFrame with vintage-level summary statistics
        """
        if self.data is None:
            raise ValueError("No data loaded.")
        
        vintage_summary = self.data.groupby(['vintage_date', 'seasoning_month']).agg({
            'loan_amount': 'sum',
            'outstanding_balance': 'sum',
            'charge_off_amount': 'sum',
            'loan_id': 'count'
        }).reset_index()
        
        vintage_summary['charge_off_rate'] = (
            vintage_summary['charge_off_amount'] / vintage_summary['outstanding_balance']
        )
        vintage_summary['avg_loan_size'] = (
            vintage_summary['loan_amount'] / vintage_summary['loan_id']
        )
        
        return vintage_summary 