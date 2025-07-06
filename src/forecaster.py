"""
Charge-off forecaster using vintage analysis and time series methods.
Combines historical vintage patterns with forward-looking projections.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from scipy import stats
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings('ignore')


class ChargeOffForecaster:
    """
    Forecasts future charge-offs using vintage analysis and time series methods.
    """
    
    def __init__(self, vintage_analyzer, data: pd.DataFrame):
        """
        Initialize the forecaster.
        
        Args:
            vintage_analyzer: VintageAnalyzer instance with fitted seasoning curves
            data: Loan performance data
        """
        self.vintage_analyzer = vintage_analyzer
        self.data = data
        self.forecast_results = None
        self.seasoning_curves = vintage_analyzer.seasoning_curves
        
    def forecast_vintage_charge_offs(self, 
                                   vintage_date: datetime,
                                   loan_amount: float,
                                   num_loans: int,
                                   forecast_horizon: int = 120,
                                   vintage_quality_adjustment: float = 1.0) -> pd.DataFrame:
        """
        Forecast charge-offs for a specific vintage.
        
        Args:
            vintage_date: Origination date of the vintage
            loan_amount: Total loan amount for the vintage
            num_loans: Number of loans in the vintage
            forecast_horizon: Number of months to forecast
            vintage_quality_adjustment: Multiplier to adjust vintage quality (1.0 = average)
            
        Returns:
            DataFrame with forecasted charge-off rates and amounts
        """
        # Get the best fitting seasoning curve
        best_curve = self._get_best_seasoning_curve()
        
        if best_curve is None:
            raise ValueError("No valid seasoning curves found. Run fit_seasoning_curves() first.")
        
        # Generate seasoning months for forecast
        seasoning_months = np.arange(0, forecast_horizon + 1)
        
        # Calculate base charge-off rates using the seasoning curve
        base_charge_off_rates = best_curve['function'](seasoning_months, *best_curve['params'])
        
        # Apply vintage quality adjustment
        adjusted_charge_off_rates = base_charge_off_rates * vintage_quality_adjustment
        
        # Calculate outstanding balances (assuming linear amortization)
        avg_loan_size = loan_amount / num_loans
        outstanding_balances = []
        
        for month in seasoning_months:
            # Simple linear amortization - in practice, use actual payment schedules
            remaining_balance = loan_amount * (1 - month / forecast_horizon)
            outstanding_balances.append(max(0, remaining_balance))
        
        # Calculate charge-off amounts
        charge_off_amounts = np.array(adjusted_charge_off_rates) * np.array(outstanding_balances)
        
        # Create forecast DataFrame
        forecast_df = pd.DataFrame({
            'vintage_date': vintage_date,
            'seasoning_month': seasoning_months,
            'outstanding_balance': outstanding_balances,
            'charge_off_rate': adjusted_charge_off_rates,
            'charge_off_amount': charge_off_amounts,
            'cumulative_charge_off_amount': np.cumsum(charge_off_amounts),
            'cumulative_charge_off_rate': np.cumsum(charge_off_amounts) / loan_amount
        })
        
        return forecast_df
    
    def forecast_portfolio_charge_offs(self, 
                                     portfolio_data: pd.DataFrame,
                                     forecast_end_date: datetime,
                                     vintage_quality_model: Optional[Dict] = None) -> pd.DataFrame:
        """
        Forecast charge-offs for an entire portfolio.
        
        Args:
            portfolio_data: DataFrame with vintage information (vintage_date, loan_amount, num_loans)
            forecast_end_date: End date for the forecast
            vintage_quality_model: Optional model to predict vintage quality adjustments
            
        Returns:
            DataFrame with portfolio-level charge-off forecasts
        """
        all_forecasts = []
        
        for _, vintage_info in portfolio_data.iterrows():
            vintage_date = vintage_info['vintage_date']
            loan_amount = vintage_info['loan_amount']
            num_loans = vintage_info['num_loans']
            
            # Determine vintage quality adjustment
            if vintage_quality_model is not None:
                vintage_quality_adjustment = self._predict_vintage_quality(
                    vintage_date, vintage_quality_model
                )
            else:
                vintage_quality_adjustment = 1.0
            
            # Calculate forecast horizon
            forecast_horizon = ((forecast_end_date.year - vintage_date.year) * 12 + 
                              forecast_end_date.month - vintage_date.month)
            
            # Forecast this vintage
            vintage_forecast = self.forecast_vintage_charge_offs(
                vintage_date=vintage_date,
                loan_amount=loan_amount,
                num_loans=num_loans,
                forecast_horizon=forecast_horizon,
                vintage_quality_adjustment=vintage_quality_adjustment
            )
            
            all_forecasts.append(vintage_forecast)
        
        # Combine all vintage forecasts
        portfolio_forecast = pd.concat(all_forecasts, ignore_index=True)
        
        # Aggregate by report date
        portfolio_forecast['report_date'] = portfolio_forecast.apply(
            lambda row: vintage_date + pd.DateOffset(months=row['seasoning_month']), axis=1
        )
        
        # Group by report date and sum
        portfolio_summary = portfolio_forecast.groupby('report_date').agg({
            'outstanding_balance': 'sum',
            'charge_off_amount': 'sum',
            'cumulative_charge_off_amount': 'sum'
        }).reset_index()
        
        portfolio_summary['charge_off_rate'] = (
            portfolio_summary['charge_off_amount'] / portfolio_summary['outstanding_balance']
        )
        
        self.forecast_results = portfolio_summary
        return portfolio_summary
    
    def _get_best_seasoning_curve(self) -> Optional[Dict]:
        """Get the best fitting seasoning curve based on R-squared."""
        if not self.seasoning_curves:
            return None
        
        best_curve = None
        best_r_squared = -1
        
        for curve_name, curve_info in self.seasoning_curves.items():
            if curve_info is not None and curve_info['r_squared'] > best_r_squared:
                best_r_squared = curve_info['r_squared']
                best_curve = curve_info
        
        return best_curve
    
    def _predict_vintage_quality(self, vintage_date: datetime, model: Dict) -> float:
        """
        Predict vintage quality adjustment based on historical patterns.
        
        Args:
            vintage_date: Vintage date to predict quality for
            model: Dictionary with vintage quality prediction model
            
        Returns:
            Vintage quality adjustment factor
        """
        # Extract features for prediction
        vintage_month = vintage_date.month
        vintage_year = vintage_date.year
        
        # Simple model based on seasonal patterns
        if 'monthly_seasonality' in model:
            monthly_effect = model['monthly_seasonality'].get(vintage_month, 1.0)
        else:
            monthly_effect = 1.0
        
        # Yearly trend effect
        if 'yearly_trends' in model:
            # Use average of recent years as baseline
            recent_years = [y for y in model['yearly_trends'].keys() if y >= vintage_year - 3]
            if recent_years:
                baseline = np.mean([model['yearly_trends'][y] for y in recent_years])
                current_trend = model['yearly_trends'].get(vintage_year, baseline)
                trend_effect = current_trend / baseline if baseline > 0 else 1.0
            else:
                trend_effect = 1.0
        else:
            trend_effect = 1.0
        
        # Combine effects
        quality_adjustment = monthly_effect * trend_effect
        
        # Ensure reasonable bounds
        quality_adjustment = max(0.5, min(2.0, quality_adjustment))
        
        return quality_adjustment
    
    def generate_scenario_forecasts(self, 
                                  base_forecast: pd.DataFrame,
                                  scenarios: Dict[str, float]) -> Dict[str, pd.DataFrame]:
        """
        Generate forecasts under different economic scenarios.
        
        Args:
            base_forecast: Base case forecast
            scenarios: Dictionary of scenario names and adjustment factors
            
        Returns:
            Dictionary of scenario forecasts
        """
        scenario_forecasts = {}
        
        for scenario_name, adjustment_factor in scenarios.items():
            scenario_forecast = base_forecast.copy()
            scenario_forecast['charge_off_amount'] *= adjustment_factor
            scenario_forecast['cumulative_charge_off_amount'] = scenario_forecast['charge_off_amount'].cumsum()
            scenario_forecast['charge_off_rate'] = (
                scenario_forecast['charge_off_amount'] / scenario_forecast['outstanding_balance']
            )
            
            scenario_forecasts[scenario_name] = scenario_forecast
        
        return scenario_forecasts
    
    def calculate_forecast_metrics(self, forecast_df: pd.DataFrame) -> Dict:
        """
        Calculate key forecast metrics.
        
        Args:
            forecast_df: Forecast DataFrame
            
        Returns:
            Dictionary with forecast metrics
        """
        metrics = {}
        
        # Total charge-offs
        metrics['total_charge_offs'] = forecast_df['charge_off_amount'].sum()
        
        # Peak charge-off rate
        metrics['peak_charge_off_rate'] = forecast_df['charge_off_rate'].max()
        metrics['peak_charge_off_month'] = forecast_df.loc[
            forecast_df['charge_off_rate'].idxmax(), 'report_date'
        ]
        
        # Average charge-off rate
        metrics['avg_charge_off_rate'] = forecast_df['charge_off_rate'].mean()
        
        # Charge-off timing
        cumulative_charge_offs = forecast_df['cumulative_charge_off_amount'].iloc[-1]
        if cumulative_charge_offs > 0:
            # Find when 50% of total charge-offs occur
            half_charge_offs = cumulative_charge_offs * 0.5
            half_point = forecast_df[forecast_df['cumulative_charge_off_amount'] >= half_charge_offs]
            if not half_point.empty:
                metrics['median_charge_off_timing'] = half_point.iloc[0]['report_date']
        
        # Vintage concentration risk
        if 'vintage_date' in forecast_df.columns:
            vintage_concentration = forecast_df.groupby('vintage_date')['charge_off_amount'].sum()
            metrics['vintage_concentration_hhi'] = ((vintage_concentration / vintage_concentration.sum()) ** 2).sum()
        
        return metrics
    
    def plot_forecast_results(self, 
                            forecast_df: pd.DataFrame,
                            scenarios: Optional[Dict[str, pd.DataFrame]] = None,
                            save_path: Optional[str] = None):
        """
        Plot forecast results.
        
        Args:
            forecast_df: Base forecast DataFrame
            scenarios: Optional dictionary of scenario forecasts
            save_path: Optional path to save the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Charge-off Forecast Dashboard', fontsize=16, fontweight='bold')
        
        # 1. Monthly charge-off amounts
        axes[0, 0].plot(forecast_df['report_date'], forecast_df['charge_off_amount'], 
                       'b-', linewidth=2, label='Base Case')
        
        if scenarios:
            colors = ['r--', 'g--', 'orange--']
            for i, (scenario_name, scenario_df) in enumerate(scenarios.items()):
                if i < len(colors):
                    axes[0, 0].plot(scenario_df['report_date'], scenario_df['charge_off_amount'],
                                   colors[i], linewidth=2, label=scenario_name)
        
        axes[0, 0].set_title('Monthly Charge-off Amounts')
        axes[0, 0].set_xlabel('Report Date')
        axes[0, 0].set_ylabel('Charge-off Amount ($)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Charge-off rates
        axes[0, 1].plot(forecast_df['report_date'], forecast_df['charge_off_rate'], 
                       'b-', linewidth=2, label='Base Case')
        
        if scenarios:
            for i, (scenario_name, scenario_df) in enumerate(scenarios.items()):
                if i < len(colors):
                    axes[0, 1].plot(scenario_df['report_date'], scenario_df['charge_off_rate'],
                                   colors[i], linewidth=2, label=scenario_name)
        
        axes[0, 1].set_title('Monthly Charge-off Rates')
        axes[0, 1].set_xlabel('Report Date')
        axes[0, 1].set_ylabel('Charge-off Rate')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. Cumulative charge-offs
        axes[1, 0].plot(forecast_df['report_date'], forecast_df['cumulative_charge_off_amount'], 
                       'b-', linewidth=2, label='Base Case')
        
        if scenarios:
            for i, (scenario_name, scenario_df) in enumerate(scenarios.items()):
                if i < len(colors):
                    axes[1, 0].plot(scenario_df['report_date'], scenario_df['cumulative_charge_off_amount'],
                                   colors[i], linewidth=2, label=scenario_name)
        
        axes[1, 0].set_title('Cumulative Charge-off Amounts')
        axes[1, 0].set_xlabel('Report Date')
        axes[1, 0].set_ylabel('Cumulative Charge-off Amount ($)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 4. Outstanding balance
        axes[1, 1].plot(forecast_df['report_date'], forecast_df['outstanding_balance'], 
                       'g-', linewidth=2, label='Outstanding Balance')
        axes[1, 1].set_title('Portfolio Outstanding Balance')
        axes[1, 1].set_xlabel('Report Date')
        axes[1, 1].set_ylabel('Outstanding Balance ($)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def export_forecast(self, 
                       forecast_df: pd.DataFrame,
                       output_path: str,
                       format: str = 'excel'):
        """
        Export forecast results to file.
        
        Args:
            forecast_df: Forecast DataFrame
            output_path: Path to save the file
            format: Output format ('excel', 'csv', 'parquet')
        """
        if format.lower() == 'excel':
            with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
                forecast_df.to_excel(writer, sheet_name='Forecast', index=False)
                
                # Add summary metrics
                metrics = self.calculate_forecast_metrics(forecast_df)
                metrics_df = pd.DataFrame(list(metrics.items()), columns=['Metric', 'Value'])
                metrics_df.to_excel(writer, sheet_name='Summary', index=False)
                
        elif format.lower() == 'csv':
            forecast_df.to_csv(output_path, index=False)
        elif format.lower() == 'parquet':
            forecast_df.to_parquet(output_path, index=False)
        else:
            raise ValueError(f"Unsupported format: {format}") 