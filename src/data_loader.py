"""
Data loader for vintage charge-off forecasting.
Handles loading and preprocessing of historical loan performance data with FICO segmentation.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple, Union
import warnings
warnings.filterwarnings('ignore')


class LoanDataLoader:
    """
    Loads and preprocesses loan performance data for vintage analysis with FICO segmentation.
    """
    
    # FICO score bands and corresponding risk grades
    FICO_BANDS = {
        '600-649': {'min': 600, 'max': 649, 'risk_grade': 5, 'label': 'Very High Risk'},
        '650-699': {'min': 650, 'max': 699, 'risk_grade': 4, 'label': 'High Risk'},
        '700-749': {'min': 700, 'max': 749, 'risk_grade': 3, 'label': 'Medium Risk'},
        '750-799': {'min': 750, 'max': 799, 'risk_grade': 2, 'label': 'Low Risk'},
        '800+': {'min': 800, 'max': 850, 'risk_grade': 1, 'label': 'Very Low Risk'}
    }
    
    def __init__(self, data_path: Optional[str] = None):
        """
        Initialize the data loader.
        
        Args:
            data_path: Path to the loan performance data file (optional)
        """
        self.data_path = data_path
        self.data = None
        
    def load_data(self, 
                  source: str = 'synthetic',
                  file_path: Optional[str] = None,
                  file_type: str = 'csv',
                  **kwargs) -> pd.DataFrame:
        """
        Load loan performance data from specified source.
        
        Args:
            source: Data source ('synthetic' or 'file')
            file_path: Path to data file (required if source='file')
            file_type: Type of file ('csv', 'excel', 'parquet')
            **kwargs: Additional arguments for file loading
            
        Returns:
            DataFrame with loan performance data
        """
        if source.lower() == 'synthetic':
            print("Generating synthetic loan performance data...")
            self.data = self._generate_synthetic_data()
        elif source.lower() == 'file':
            if file_path is None:
                raise ValueError("file_path is required when source='file'")
            print(f"Loading loan performance data from {file_path}...")
            self.data = self._load_from_file(file_path, file_type, **kwargs)
        else:
            raise ValueError(f"Invalid source: {source}. Use 'synthetic' or 'file'")
        
        return self.data
    
    def _generate_synthetic_data(self) -> pd.DataFrame:
        """
        Generate synthetic loan performance data with FICO segmentation.
        
        Returns:
            DataFrame with synthetic loan performance data
        """
        np.random.seed(42)  # For reproducible results
        
        # Create date range
        start_date = pd.Timestamp('2014-01-01')
        end_date = pd.Timestamp('2024-12-31')
        date_range = pd.date_range(start=start_date, end=end_date, freq='M')
        
        # Generate sample loan vintages (origination periods)
        vintage_periods = pd.date_range(start='2014-01-01', end='2023-12-31', freq='M')
        
        data_rows = []
        
        for vintage_date in vintage_periods:
            # Generate loans for each FICO band with different volumes and characteristics
            for fico_band, band_info in self.FICO_BANDS.items():
                # Number of loans varies by risk grade (more loans in middle bands)
                base_volume = {
                    5: 800,   # Very High Risk - fewer loans
                    4: 1200,  # High Risk - moderate volume
                    3: 1500,  # Medium Risk - highest volume
                    2: 1000,  # Low Risk - moderate volume
                    1: 500    # Very Low Risk - fewer loans
                }
                
                num_loans = np.random.randint(
                    int(base_volume[band_info['risk_grade']] * 0.8),
                    int(base_volume[band_info['risk_grade']] * 1.2)
                )
                
                # Generate loan IDs for this vintage and FICO band
                loan_ids = [f"LOAN_{vintage_date.strftime('%Y%m')}_{fico_band}_{i:06d}" 
                           for i in range(num_loans)]
                
                # Loan characteristics vary by FICO band
                risk_grade = band_info['risk_grade']
                
                # Loan amounts: higher FICO = larger loans
                avg_loan_size = {
                    5: 15000,   # Very High Risk - smaller loans
                    4: 25000,   # High Risk - moderate loans
                    3: 35000,   # Medium Risk - standard loans
                    2: 45000,   # Low Risk - larger loans
                    1: 60000    # Very Low Risk - largest loans
                }
                
                loan_amounts = np.random.lognormal(
                    mean=np.log(avg_loan_size[risk_grade]),
                    sigma=0.3,
                    size=num_loans
                )
                
                # Interest rates: higher FICO = lower rates
                base_rates = {
                    5: 0.15,  # Very High Risk - highest rates
                    4: 0.12,  # High Risk - high rates
                    3: 0.09,  # Medium Risk - moderate rates
                    2: 0.07,  # Low Risk - low rates
                    1: 0.05   # Very Low Risk - lowest rates
                }
                
                interest_rates = np.random.normal(
                    mean=base_rates[risk_grade],
                    std=0.02,
                    size=num_loans
                )
                
                # Terms: higher FICO = longer terms
                term_options = {
                    5: [60, 84],      # Very High Risk - shorter terms
                    4: [84, 120],     # High Risk - moderate terms
                    3: [120, 180],    # Medium Risk - standard terms
                    2: [180, 240],    # Low Risk - longer terms
                    1: [240, 300]     # Very Low Risk - longest terms
                }
                
                terms = np.random.choice(
                    term_options[risk_grade],
                    size=num_loans
                )
                
                # Generate FICO scores within the band
                fico_scores = np.random.randint(
                    band_info['min'],
                    band_info['max'] + 1,
                    size=num_loans
                )
                
                # Generate performance data for each loan over time
                for i, loan_id in enumerate(loan_ids):
                    loan_amount = loan_amounts[i]
                    interest_rate = interest_rates[i]
                    term = terms[i]
                    fico_score = fico_scores[i]
                    
                    # Calculate seasoning months for this loan
                    for report_date in date_range:
                        if report_date >= vintage_date:
                            seasoning_month = ((report_date.year - vintage_date.year) * 12 + 
                                             report_date.month - vintage_date.month)
                            
                            if seasoning_month <= term:
                                # Calculate outstanding balance
                                remaining_balance = loan_amount * (1 - seasoning_month / term)
                                
                                # Calculate charge-off rate based on vintage, seasoning, and FICO
                                base_rate = self._get_base_charge_off_rate(risk_grade)
                                
                                # Vintage effect (some vintages perform better/worse)
                                vintage_effect = 0.5 + 0.5 * np.sin(vintage_date.month / 12 * 2 * np.pi)
                                
                                # Seasoning effect (charge-offs typically peak around 18-24 months)
                                seasoning_effect = np.exp(-((seasoning_month - 20) ** 2) / 100)
                                
                                # FICO effect (higher FICO = lower charge-off rates)
                                fico_effect = 1.0 - (risk_grade - 1) * 0.15
                                
                                # Economic cycle effect (simulate 2008-like crisis)
                                if 2018 <= vintage_date.year <= 2020:
                                    crisis_effect = 1.5
                                else:
                                    crisis_effect = 1.0
                                
                                # Random variation
                                random_effect = np.random.normal(1, 0.1)
                                
                                charge_off_rate = (base_rate * vintage_effect * seasoning_effect * 
                                                 fico_effect * crisis_effect * random_effect)
                                charge_off_rate = max(0, min(0.25, charge_off_rate))  # Cap at 25%
                                
                                data_rows.append({
                                    'loan_id': loan_id,
                                    'vintage_date': vintage_date,
                                    'report_date': report_date,
                                    'seasoning_month': seasoning_month,
                                    'fico_score': fico_score,
                                    'fico_band': fico_band,
                                    'risk_grade': risk_grade,
                                    'loan_amount': loan_amount,
                                    'interest_rate': interest_rate,
                                    'term': term,
                                    'outstanding_balance': remaining_balance,
                                    'charge_off_rate': charge_off_rate,
                                    'charge_off_amount': remaining_balance * charge_off_rate
                                })
        
        self.data = pd.DataFrame(data_rows)
        return self.data
    
    def _get_base_charge_off_rate(self, risk_grade: int) -> float:
        """
        Get base charge-off rate for a given risk grade.
        
        Args:
            risk_grade: Risk grade (1-5)
            
        Returns:
            Base charge-off rate
        """
        base_rates = {
            5: 0.08,  # Very High Risk - 8% base rate
            4: 0.06,  # High Risk - 6% base rate
            3: 0.04,  # Medium Risk - 4% base rate
            2: 0.025, # Low Risk - 2.5% base rate
            1: 0.015  # Very Low Risk - 1.5% base rate
        }
        return base_rates.get(risk_grade, 0.04)
    
    def _load_from_file(self, file_path: str, file_type: str = 'csv', **kwargs) -> pd.DataFrame:
        """
        Load data from file with validation for required columns.
        
        Args:
            file_path: Path to the data file
            file_type: Type of file ('csv', 'excel', 'parquet')
            **kwargs: Additional arguments for file loading
            
        Returns:
            DataFrame with loan performance data
        """
        if file_type.lower() == 'csv':
            self.data = pd.read_csv(file_path, **kwargs)
        elif file_type.lower() == 'excel':
            self.data = pd.read_excel(file_path, **kwargs)
        elif file_type.lower() == 'parquet':
            self.data = pd.read_parquet(file_path, **kwargs)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
        
        # Validate required columns
        required_columns = [
            'loan_id', 'vintage_date', 'report_date', 'seasoning_month',
            'fico_score', 'loan_amount', 'charge_off_rate', 'charge_off_amount'
        ]
        
        missing_columns = [col for col in required_columns if col not in self.data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Add FICO band and risk grade if not present
        if 'fico_band' not in self.data.columns:
            self.data['fico_band'] = self.data['fico_score'].apply(self._assign_fico_band)
        
        if 'risk_grade' not in self.data.columns:
            self.data['risk_grade'] = self.data['fico_band'].apply(
                lambda x: self.FICO_BANDS[x]['risk_grade']
            )
        
        return self.data
    
    def _assign_fico_band(self, fico_score: int) -> str:
        """
        Assign FICO score to appropriate band.
        
        Args:
            fico_score: FICO score
            
        Returns:
            FICO band string
        """
        for band, info in self.FICO_BANDS.items():
            if info['min'] <= fico_score <= info['max']:
                return band
        return '800+'  # Default for scores above 850
    
    def preprocess_data(self) -> pd.DataFrame:
        """
        Preprocess the loaded data for vintage analysis.
        
        Returns:
            Preprocessed DataFrame
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        # Convert date columns
        date_columns = ['vintage_date', 'report_date']
        for col in date_columns:
            if col in self.data.columns:
                self.data[col] = pd.to_datetime(self.data[col])
        
        # Sort by vintage_date, fico_band, and report_date
        self.data = self.data.sort_values(['vintage_date', 'fico_band', 'report_date'])
        
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
    
    def get_vintage_summary_by_fico(self) -> pd.DataFrame:
        """
        Create vintage summary statistics by FICO band.
        
        Returns:
            DataFrame with vintage-level summary statistics by FICO band
        """
        if self.data is None:
            raise ValueError("No data loaded.")
        
        vintage_summary = self.data.groupby(['vintage_date', 'fico_band', 'seasoning_month']).agg({
            'loan_amount': 'sum',
            'outstanding_balance': 'sum',
            'charge_off_amount': 'sum',
            'loan_id': 'count'
        }).reset_index()
        
        vintage_summary['charge_off_rate'] = (
            vintage_summary['charge_off_amount'] / vintage_summary['outstanding_balance']
        )
        vintage_summary['cumulative_charge_off_rate'] = (
            vintage_summary.groupby(['vintage_date', 'fico_band'])['charge_off_amount'].cumsum() /
            vintage_summary.groupby(['vintage_date', 'fico_band'])['loan_amount'].first()
        )
        
        vintage_summary['avg_loan_size'] = (
            vintage_summary['loan_amount'] / vintage_summary['loan_id']
        )
        
        # Add risk grade
        vintage_summary['risk_grade'] = vintage_summary['fico_band'].apply(
            lambda x: self.FICO_BANDS[x]['risk_grade']
        )
        
        return vintage_summary
    
    def get_fico_mix_by_vintage(self) -> pd.DataFrame:
        """
        Get FICO band mix for each vintage.
        
        Returns:
            DataFrame with FICO band distribution by vintage
        """
        if self.data is None:
            raise ValueError("No data loaded.")
        
        # Get unique loans by vintage and FICO band
        loan_mix = self.data.groupby(['vintage_date', 'fico_band', 'loan_id']).first().reset_index()
        
        # Calculate mix by vintage
        vintage_mix = loan_mix.groupby(['vintage_date', 'fico_band']).agg({
            'loan_amount': 'sum',
            'loan_id': 'count'
        }).reset_index()
        
        # Calculate percentages
        vintage_totals = vintage_mix.groupby('vintage_date').agg({
            'loan_amount': 'sum',
            'loan_id': 'sum'
        }).reset_index()
        
        vintage_mix = vintage_mix.merge(
            vintage_totals,
            on='vintage_date',
            suffixes=('', '_total')
        )
        
        vintage_mix['amount_pct'] = vintage_mix['loan_amount'] / vintage_mix['loan_amount_total']
        vintage_mix['count_pct'] = vintage_mix['loan_id'] / vintage_mix['loan_id_total']
        
        # Add risk grade
        vintage_mix['risk_grade'] = vintage_mix['fico_band'].apply(
            lambda x: self.FICO_BANDS[x]['risk_grade']
        )
        
        return vintage_mix
    
    def get_data_summary(self) -> Dict:
        """
        Get comprehensive summary of the loaded data.
        
        Returns:
            Dictionary with data summary statistics
        """
        if self.data is None:
            raise ValueError("No data loaded.")
        
        summary = {
            'total_records': len(self.data),
            'total_loans': self.data['loan_id'].nunique(),
            'total_vintages': self.data['vintage_date'].nunique(),
            'date_range': {
                'start': self.data['vintage_date'].min(),
                'end': self.data['vintage_date'].max()
            },
            'total_loan_amount': self.data['loan_amount'].sum(),
            'fico_bands': self.data['fico_band'].value_counts().to_dict(),
            'risk_grades': self.data['risk_grade'].value_counts().sort_index().to_dict(),
            'avg_charge_off_rate': self.data['charge_off_rate'].mean(),
            'avg_fico_score': self.data['fico_score'].mean()
        }
        
        return summary 