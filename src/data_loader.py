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
                                
                                # Enforce charge-off policy: no charge-off before month 6
                                if seasoning_month < 6:
                                    charge_off_flag = 0
                                else:
                                    # Calculate charge-off flag based on vintage, seasoning, and FICO
                                    base_rate = self._get_base_charge_off_rate(risk_grade)
                                    vintage_effect = 0.5 + 0.5 * np.sin(vintage_date.month / 12 * 2 * np.pi)
                                    seasoning_effect = np.exp(-((seasoning_month - 20) ** 2) / 100)
                                    fico_effect = 1.0 - (risk_grade - 1) * 0.15
                                    if 2018 <= vintage_date.year <= 2020:
                                        crisis_effect = 1.5
                                    else:
                                        crisis_effect = 1.0
                                    random_effect = np.random.normal(1, 0.1)
                                    charge_off_flag = (base_rate * vintage_effect * seasoning_effect * 
                                                     fico_effect * crisis_effect * random_effect)
                                    # Convert to binary flag (simulate actual charge-off event)
                                    charge_off_flag = 1 if np.random.rand() < charge_off_flag else 0
                                
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
                                    'charge_off_flag': charge_off_flag,
                                    'charge_off_amount': remaining_balance * charge_off_flag
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
        
        Expected file format:
        
        Required Columns:
        - loan_id (str): Unique loan identifier
        - vintage_date (date): Loan origination date (YYYY-MM-DD format)
        - report_date (date): Performance reporting date (YYYY-MM-DD format)
        - seasoning_month (int): Months since origination (0, 1, 2, ...)
        - fico_score (int): FICO score at origination (300-850)
        - loan_amount (float): Original loan amount
        - charge_off_flag (int): Binary charge-off flag (0 = no charge-off, 1 = charged off)
        - charge_off_amount (float): Amount charged off (0 if no charge-off, typically 70-90% of original principal if defaulted)
        
        Optional Columns:
        - outstanding_balance (float): Outstanding balance at report date
        - interest_rate (float): Loan interest rate (0-1, e.g., 0.085 = 8.5%)
        - term (int): Loan term in months
        - fico_band (str): FICO score band (auto-assigned if missing)
        - risk_grade (int): Risk grade 1-5 (auto-assigned if missing)
        
        Example CSV format:
        loan_id,vintage_date,report_date,seasoning_month,fico_score,loan_amount,charge_off_flag,charge_off_amount,outstanding_balance
        LOAN_001,2014-01-15,2014-02-15,1,650,25000.00,0,0.00,24875.00
        LOAN_001,2014-01-15,2014-03-15,2,650,25000.00,0,0.00,24800.37
        LOAN_001,2014-01-15,2014-04-15,3,650,25000.00,1,17500.00,0.00
        
        Args:
            file_path: Path to the data file
            file_type: Type of file ('csv', 'excel', 'parquet')
            **kwargs: Additional arguments for file loading
            
        Returns:
            DataFrame with loan performance data
            
        Raises:
            ValueError: If required columns are missing or data validation fails
        """
        # Load data based on file type
        if file_type.lower() == 'csv':
            self.data = pd.read_csv(file_path, **kwargs)
        elif file_type.lower() == 'excel':
            self.data = pd.read_excel(file_path, **kwargs)
        elif file_type.lower() == 'parquet':
            self.data = pd.read_parquet(file_path, **kwargs)
        else:
            raise ValueError(f"Unsupported file type: {file_type}. Use 'csv', 'excel', or 'parquet'")
        
        # Validate required columns
        required_columns = [
            'loan_id', 'vintage_date', 'report_date', 'seasoning_month',
            'fico_score', 'loan_amount', 'charge_off_flag', 'charge_off_amount'
        ]
        
        missing_columns = [col for col in required_columns if col not in self.data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}\n"
                           f"Required columns: {required_columns}\n"
                           f"Available columns: {list(self.data.columns)}")
        
        # Validate data types and ranges
        self._validate_data_types()
        self._validate_data_ranges()
        
        # Add FICO band and risk grade if not present
        if 'fico_band' not in self.data.columns:
            self.data['fico_band'] = self.data['fico_score'].apply(self._assign_fico_band)
        
        if 'risk_grade' not in self.data.columns:
            self.data['risk_grade'] = self.data['fico_band'].apply(
                lambda x: self.FICO_BANDS[x]['risk_grade']
            )
        
        return self.data
    
    def _validate_data_types(self):
        """Validate data types of loaded data."""
        # Convert date columns
        for date_col in ['vintage_date', 'report_date']:
            if date_col in self.data.columns:
                try:
                    self.data[date_col] = pd.to_datetime(self.data[date_col])
                except Exception as e:
                    raise ValueError(f"Invalid date format in {date_col}: {e}")
        
        # Validate numeric columns
        numeric_columns = {
            'seasoning_month': int,
            'fico_score': int,
            'loan_amount': float,
            'charge_off_flag': int,
            'charge_off_amount': float
        }
        
        for col, expected_type in numeric_columns.items():
            if col in self.data.columns:
                try:
                    if expected_type == int:
                        self.data[col] = self.data[col].astype(int)
                    else:
                        self.data[col] = self.data[col].astype(float)
                except Exception as e:
                    raise ValueError(f"Invalid data type in {col}: expected {expected_type}, got {e}")
    
    def _validate_data_ranges(self):
        """Validate data ranges and business logic."""
        # Validate FICO scores
        if 'fico_score' in self.data.columns:
            invalid_fico = self.data[
                (self.data['fico_score'] < 300) | (self.data['fico_score'] > 850)
            ]
            if not invalid_fico.empty:
                raise ValueError(f"Invalid FICO scores found: {invalid_fico['fico_score'].unique()}\n"
                               f"FICO scores must be between 300 and 850")
        
        # Validate charge-off flags (binary flags)
        if 'charge_off_flag' in self.data.columns:
            invalid_flags = self.data[
                ~self.data['charge_off_flag'].isin([0, 1])
            ]
            if not invalid_flags.empty:
                raise ValueError(f"Invalid charge-off flags found: {invalid_flags['charge_off_flag'].unique()}\n"
                               f"Charge-off flags must be 0 or 1 (binary flags)")
        
        # Validate seasoning months
        if 'seasoning_month' in self.data.columns:
            invalid_seasoning = self.data[self.data['seasoning_month'] < 0]
            if not invalid_seasoning.empty:
                raise ValueError(f"Invalid seasoning months found: {invalid_seasoning['seasoning_month'].unique()}\n"
                               f"Seasoning months must be non-negative")
        
        # Validate loan amounts
        if 'loan_amount' in self.data.columns:
            invalid_amounts = self.data[self.data['loan_amount'] <= 0]
            if not invalid_amounts.empty:
                raise ValueError(f"Invalid loan amounts found: {invalid_amounts['loan_amount'].unique()}\n"
                               f"Loan amounts must be positive")
        
        # Validate charge-off amounts
        if 'charge_off_amount' in self.data.columns:
            invalid_charge_offs = self.data[self.data['charge_off_amount'] < 0]
            if not invalid_charge_offs.empty:
                raise ValueError(f"Invalid charge-off amounts found: {invalid_charge_offs['charge_off_amount'].unique()}\n"
                               f"Charge-off amounts must be non-negative")
    
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
        
        # Handle incomplete vintage data (fill missing seasoning months for charged-off loans)
        self.data = self._complete_vintage_data()
        
        # Reassign early charge-offs to month 6 for modeling
        mask_early_co = (self.data['charge_off_flag'] == 1) & (self.data['seasoning_month'] < 6)
        if mask_early_co.any():
            print(f"Reassigning {mask_early_co.sum()} early charge-offs to seasoning_month 6 for modeling.")
            self.data.loc[mask_early_co, 'seasoning_month'] = 6
        
        # Flag and report early charge-offs
        early_chargeoffs = self.data[(self.data['charge_off_flag'] == 1) & (self.data['seasoning_month'] < 6)]
        if not early_chargeoffs.empty:
            print(f"Warning: {len(early_chargeoffs)} loans charged off before 6 months. These are excluded from model fitting.")
            print(early_chargeoffs[['loan_id', 'vintage_date', 'seasoning_month', 'charge_off_flag']].head())

        return self.data
    
    def _complete_vintage_data(self) -> pd.DataFrame:
        """
        Complete vintage data by filling in missing seasoning months for charged-off loans.
        This handles cases where loans disappear from the data after charge-off.
        
        Returns:
            DataFrame with completed vintage data
        """
        print("   Completing vintage data (filling missing seasoning months for charged-off loans)...")
        
        # Get unique vintage-FICO combinations
        vintage_fico_combinations = self.data.groupby(['vintage_date', 'fico_band']).size().reset_index()
        
        completed_data = []
        
        for _, row in vintage_fico_combinations.iterrows():
            vintage_date = row['vintage_date']
            fico_band = row['fico_band']
            
            # Get data for this vintage-FICO combination
            vintage_data = self.data[
                (self.data['vintage_date'] == vintage_date) & 
                (self.data['fico_band'] == fico_band)
            ].copy()
            
            # Find the maximum seasoning month in the data
            max_seasoning = vintage_data['seasoning_month'].max()
            
            # Check if any loans in this vintage charged off
            charged_off_loans = vintage_data[vintage_data['charge_off_flag'] == 1]
            
            if not charged_off_loans.empty:
                # Find the first seasoning month where any loan charged off
                first_charge_off_month = charged_off_loans['seasoning_month'].min()
                
                # Get loan characteristics from the first record
                sample_record = vintage_data.iloc[0]
                loan_id = sample_record['loan_id']
                fico_score = sample_record['fico_score']
                loan_amount = sample_record['loan_amount']
                interest_rate = sample_record.get('interest_rate', 0.08)
                term = sample_record.get('term', 120)
                
                # Create complete seasoning curve (up to max seasoning or term)
                max_months = max(max_seasoning, term)
                
                for month in range(1, max_months + 1):
                    existing_record = vintage_data[vintage_data['seasoning_month'] == month]
                    
                    if not existing_record.empty:
                        # Use existing record
                        completed_data.append(existing_record.iloc[0].to_dict())
                    else:
                        # Create missing record for charged-off loan
                        if month >= first_charge_off_month:
                            # Loan is charged off
                            charge_off_flag = 1
                            charge_off_amount = 0.0  # Already charged off in previous month
                            outstanding_balance = 0.0
                        else:
                            # Loan is still performing (shouldn't happen, but handle gracefully)
                            charge_off_flag = 0
                            charge_off_amount = 0.0
                            # Estimate outstanding balance
                            outstanding_balance = loan_amount * (1 - month / term)
                        
                        # Calculate report date
                        report_date = vintage_date + pd.DateOffset(months=month)
                        
                        new_record = {
                            'loan_id': loan_id,
                            'vintage_date': vintage_date,
                            'report_date': report_date,
                            'seasoning_month': month,
                            'fico_score': fico_score,
                            'fico_band': fico_band,
                            'risk_grade': sample_record.get('risk_grade', self.FICO_BANDS[fico_band]['risk_grade']),
                            'loan_amount': loan_amount,
                            'interest_rate': interest_rate,
                            'term': term,
                            'outstanding_balance': outstanding_balance,
                            'charge_off_flag': charge_off_flag,
                            'charge_off_amount': charge_off_amount,
                            'vintage_year': vintage_date.year,
                            'vintage_month': vintage_date.month,
                            'report_year': report_date.year,
                            'report_month': report_date.month,
                            'vintage_age_months': month
                        }
                        completed_data.append(new_record)
            else:
                # No charge-offs in this vintage, use existing data as-is
                completed_data.extend(vintage_data.to_dict('records'))
        
        # Convert back to DataFrame
        completed_df = pd.DataFrame(completed_data)
        
        # Sort by vintage_date, fico_band, and seasoning_month
        completed_df = completed_df.sort_values(['vintage_date', 'fico_band', 'seasoning_month'])
        
        # Update the data
        self.data = completed_df
        
        print(f"   Completed vintage data: {len(self.data)} records")
        
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
        
        vintage_summary['charge_off_flag'] = (
            vintage_summary['charge_off_amount'] / vintage_summary['outstanding_balance']
        )
        vintage_summary['cumulative_charge_off_flag'] = (
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
            'avg_charge_off_flag': self.data['charge_off_flag'].mean(),
            'avg_fico_score': self.data['fico_score'].mean()
        }
        
        return summary
    
    @staticmethod
    def create_sample_template(file_path: str = 'loan_data_template.csv', file_type: str = 'csv'):
        """
        Create a sample template file in the correct format for real data input.
        Args:
            file_path: Path to save the template file
            file_type: 'csv', 'excel', or 'parquet'
        """
        import pandas as pd
        sample_data = [
            {
                'loan_id': 'LOAN_001',
                'vintage_date': '2014-01-15',
                'report_date': '2014-02-15',
                'seasoning_month': 1,
                'fico_score': 650,
                'loan_amount': 25000.00,
                'charge_off_flag': 0,
                'charge_off_amount': 0.00,
                'outstanding_balance': 24875.00,
                'interest_rate': 0.085,
                'term': 120
            },
            {
                'loan_id': 'LOAN_001',
                'vintage_date': '2014-01-15',
                'report_date': '2014-03-15',
                'seasoning_month': 2,
                'fico_score': 650,
                'loan_amount': 25000.00,
                'charge_off_flag': 0,
                'charge_off_amount': 0.00,
                'outstanding_balance': 24800.37,
                'interest_rate': 0.085,
                'term': 120
            },
            {
                'loan_id': 'LOAN_001',
                'vintage_date': '2014-01-15',
                'report_date': '2014-04-15',
                'seasoning_month': 3,
                'fico_score': 650,
                'loan_amount': 25000.00,
                'charge_off_flag': 1,
                'charge_off_amount': 17500.00,
                'outstanding_balance': 0.00,
                'interest_rate': 0.085,
                'term': 120
            },
            {
                'loan_id': 'LOAN_002',
                'vintage_date': '2014-01-15',
                'report_date': '2014-02-15',
                'seasoning_month': 1,
                'fico_score': 720,
                'loan_amount': 35000.00,
                'charge_off_flag': 0,
                'charge_off_amount': 0.00,
                'outstanding_balance': 34947.50,
                'interest_rate': 0.075,
                'term': 180
            }
        ]
        df = pd.DataFrame(sample_data)
        if file_type == 'csv':
            df.to_csv(file_path, index=False)
        elif file_type == 'excel':
            df.to_excel(file_path, index=False)
        elif file_type == 'parquet':
            df.to_parquet(file_path, index=False)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
        print(f"Sample template saved to {file_path}")

    def generate_synthetic_data(self, 
                               num_vintages: int = 12,
                               loans_per_vintage: int = 100,
                               max_seasoning: int = 36,
                               incomplete_vintages: bool = False) -> pd.DataFrame:
        """
        Generate synthetic vintage data for testing and demonstration.
        
        Args:
            num_vintages: Number of vintage months to generate
            loans_per_vintage: Number of loans per vintage
            max_seasoning: Maximum seasoning months to generate
            incomplete_vintages: If True, loans disappear after charge-off
            
        Returns:
            DataFrame with synthetic vintage data
        """
        print(f"Generating synthetic vintage data...")
        print(f"  Vintages: {num_vintages}")
        print(f"  Loans per vintage: {loans_per_vintage}")
        print(f"  Max seasoning: {max_seasoning}")
        print(f"  Incomplete vintages: {incomplete_vintages}")
        
        data = []
        start_date = pd.Timestamp('2020-01-01')
        
        for vintage_idx in range(num_vintages):
            vintage_date = start_date + pd.DateOffset(months=vintage_idx)
            
            # Generate loans for this vintage with FICO distribution
            for loan_idx in range(loans_per_vintage):
                # Assign FICO band based on distribution
                fico_band = np.random.choice(
                    list(self.FICO_BANDS.keys()),
                    p=[0.15, 0.25, 0.30, 0.20, 0.10]  # Distribution across bands
                )
                
                fico_config = self.FICO_BANDS[fico_band]
                fico_score = np.random.randint(fico_config['min_score'], fico_config['max_score'] + 1)
                risk_grade = fico_config['risk_grade']
                
                # Loan characteristics
                loan_amount = np.random.lognormal(10.5, 0.3)  # Mean ~$36K
                interest_rate = fico_config['base_rate'] + np.random.normal(0, 0.02)
                term = np.random.choice([60, 84, 120], p=[0.3, 0.4, 0.3])
                
                # Charge-off probability based on FICO band
                charge_off_prob = fico_config['charge_off_prob']
                
                # Generate seasoning data
                loan_charged_off = False
                charge_off_month = None
                
                for seasoning_month in range(1, max_seasoning + 1):
                    report_date = vintage_date + pd.DateOffset(months=seasoning_month)
                    
                    # Calculate outstanding balance
                    if seasoning_month <= term:
                        outstanding_balance = loan_amount * (1 - seasoning_month / term)
                    else:
                        outstanding_balance = 0.0
                    
                    # Determine if loan charges off this month
                    charge_off_flag = 0
                    charge_off_amount = 0.0
                    
                    if not loan_charged_off and outstanding_balance > 0:
                        # Monthly charge-off probability increases with seasoning
                        monthly_co_prob = charge_off_prob * (1 + seasoning_month / 12)
                        
                        if np.random.random() < monthly_co_prob:
                            loan_charged_off = True
                            charge_off_month = seasoning_month
                            charge_off_flag = 1
                            charge_off_amount = outstanding_balance * 0.8  # 80% of balance
                            outstanding_balance = 0.0
                    
                    # For incomplete vintages, stop generating records after charge-off
                    if incomplete_vintages and loan_charged_off:
                        # Only include the charge-off month record
                        if seasoning_month == charge_off_month:
                            record = {
                                'loan_id': f"L{vintage_idx:03d}_{loan_idx:04d}",
                                'vintage_date': vintage_date,
                                'report_date': report_date,
                                'seasoning_month': seasoning_month,
                                'fico_score': fico_score,
                                'fico_band': fico_band,
                                'risk_grade': risk_grade,
                                'loan_amount': loan_amount,
                                'interest_rate': interest_rate,
                                'term': term,
                                'outstanding_balance': outstanding_balance,
                                'charge_off_flag': charge_off_flag,
                                'charge_off_amount': charge_off_amount
                            }
                            data.append(record)
                        break
                    else:
                        # For complete vintages, continue generating records
                        record = {
                            'loan_id': f"L{vintage_idx:03d}_{loan_idx:04d}",
                            'vintage_date': vintage_date,
                            'report_date': report_date,
                            'seasoning_month': seasoning_month,
                            'fico_score': fico_score,
                            'fico_band': fico_band,
                            'risk_grade': risk_grade,
                            'loan_amount': loan_amount,
                            'interest_rate': interest_rate,
                            'term': term,
                            'outstanding_balance': outstanding_balance,
                            'charge_off_flag': charge_off_flag,
                            'charge_off_amount': charge_off_amount
                        }
                        data.append(record)
        
        df = pd.DataFrame(data)
        print(f"Generated {len(df)} records")
        
        return df 