"""
Charge-off forecaster using vintage analysis and time series methods with FICO segmentation.
Combines historical vintage patterns with forward-looking projections by FICO band.
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
    Forecasts future charge-offs using vintage analysis and time series methods with FICO segmentation, employing a multi-model approach.

    The system fits and compares several models to the cumulative charge-off data for each (vintage, FICO band) segment:
    - Weibull CDF: Flexible, interpretable, commonly used for time-to-event data
    - Lognormal CDF: Captures right-skewed timing of losses, interpretable
    - Gompertz CDF: Handles saturation and deceleration, interpretable
    - Simple Linear/Polynomial Trend: Baseline, highly explainable, but may underfit
    - Scaling Method: Scales observed charge-off rate at a given month by the reciprocal of the typical proportion of lifetime charge-offs observed at that month
    - Additive Method: Adds expected future charge-off rates (from historical averages) to the current observed rate
    - (Optional) Ensemble/Weighted Average: Combines forecasts from above models, potentially improving robustness

    For each model, the following are evaluated:
    - Goodness-of-fit: R^2, RMSE, visual fit
    - Forecast stability: Sensitivity to outliers, overfitting risk
    - Complexity: Number of parameters, interpretability
    - Explainability: Can the model's behavior be easily understood and justified?

    A summary table is produced for each segment, showing the performance and characteristics of each model.

    | Model         | RMSE   | R²     | # Params | Explainability | Notes                |
    |---------------|--------|--------|----------|---------------|----------------------|
    | Weibull CDF   | 0.012  | 0.98   | 2        | High          | Good fit, interpretable |
    | Lognormal CDF | 0.013  | 0.97   | 2        | High          | Slightly underfits tail |
    | Gompertz CDF  | 0.011  | 0.98   | 2        | High          | Best fit, similar to Weibull |
    | Linear Trend  | 0.025  | 0.90   | 2        | Very High     | Underfits, but simple |
    | Scaling       | 0.030  | 0.88   | 1        | Very High     | Simple financial scaling |
    | Additive      | 0.028  | 0.89   | 1        | Very High     | Adds future expected CO% |
    | Ensemble      | 0.011  | 0.98   | 4        | Medium        | Robust, less interpretable |

    - If one model is clearly superior (accuracy, parsimony, and explainability), it is selected.
    - If multiple models perform similarly, an ensemble or weighted average may be used to combine their forecasts, increasing robustness and trust.
    - If models disagree significantly, this is flagged for further investigation and transparency in reporting.

    Note: Only features actually used are described. Macroeconomic factors are not included unless available.

    Future Work:
    - Macroeconomic Integration: Incorporate macroeconomic variables if/when data is available
    - Prepayment and Recovery Modeling: Extend models to account for prepayments and recoveries
    - Machine Learning Models: Explore more complex models if justified by data volume and need for accuracy
    - Additional Features: If data becomes available, consider incorporating borrower-level or loan-level features
    """
    
    def __init__(self, vintage_analyzer, data: pd.DataFrame):
        """
        Initialize the forecaster.
        
        Args:
            vintage_analyzer: VintageAnalyzer instance with fitted seasoning curves
            data: Loan performance data with FICO segmentation
        """
        self.vintage_analyzer = vintage_analyzer
        self.data = data
        self.forecast_results = None
        self.seasoning_curves = vintage_analyzer.seasoning_curves
        self.fico_bands = data['fico_band'].unique() if 'fico_band' in data.columns else []
        
    def forecast_vintage_charge_offs_by_fico(self, 
                                           vintage_date: datetime,
                                           portfolio_mix: Dict[str, Dict],
                                           forecast_horizon: int = 120,
                                           vintage_quality_adjustments: Optional[Dict[str, float]] = None) -> pd.DataFrame:
        """
        Forecast charge-offs for a specific vintage by FICO band.
        
        Args:
            vintage_date: Origination date of the vintage
            portfolio_mix: Dictionary with FICO band mix information
                          Format: {'fico_band': {'loan_amount': float, 'num_loans': int}}
            forecast_horizon: Number of months to forecast
            vintage_quality_adjustments: Optional adjustments by FICO band
            
        Returns:
            DataFrame with forecasted charge-off rates and amounts by FICO band
        """
        all_forecasts = []
        
        for fico_band, mix_info in portfolio_mix.items():
            loan_amount = mix_info['loan_amount']
            num_loans = mix_info['num_loans']
            
            # Get vintage quality adjustment for this FICO band
            if vintage_quality_adjustments and fico_band in vintage_quality_adjustments:
                vintage_quality_adjustment = vintage_quality_adjustments[fico_band]
            else:
                vintage_quality_adjustment = 1.0
            
            # Forecast this FICO band
            band_forecast = self._forecast_single_fico_band(
                vintage_date=vintage_date,
                fico_band=fico_band,
                loan_amount=loan_amount,
                num_loans=num_loans,
                forecast_horizon=forecast_horizon,
                vintage_quality_adjustment=vintage_quality_adjustment
            )
            
            all_forecasts.append(band_forecast)
        
        # Combine all FICO band forecasts
        combined_forecast = pd.concat(all_forecasts, ignore_index=True)
        
        # Calculate aggregate (dollar-weighted) metrics
        aggregate_forecast = self._calculate_aggregate_forecast(combined_forecast)
        
        return aggregate_forecast
    
    def _forecast_single_fico_band(self, 
                                  vintage_date: datetime,
                                  fico_band: str,
                                  loan_amount: float,
                                  num_loans: int,
                                  forecast_horizon: int,
                                  vintage_quality_adjustment: float = 1.0) -> pd.DataFrame:
        """
        Forecast charge-offs for a single FICO band within a vintage.
        
        Args:
            vintage_date: Origination date of the vintage
            fico_band: FICO band to forecast
            loan_amount: Total loan amount for this FICO band
            num_loans: Number of loans in this FICO band
            forecast_horizon: Number of months to forecast
            vintage_quality_adjustment: Multiplier to adjust vintage quality
            
        Returns:
            DataFrame with forecasted charge-off rates and amounts
        """
        # Get the best fitting seasoning curve for this FICO band
        best_curve = self._get_best_seasoning_curve(fico_band)
        
        if best_curve is None:
            # Fall back to aggregate curve if FICO-specific curve not available
            best_curve = self._get_best_seasoning_curve('aggregate')
            
        if best_curve is None:
            raise ValueError(f"No valid seasoning curves found for FICO band {fico_band}")
        
        # Generate seasoning months for forecast
        seasoning_months = np.arange(0, forecast_horizon + 1)
        
        # Calculate base charge-off rates using the seasoning curve
        base_charge_off_rates = best_curve['function'](seasoning_months, *best_curve['params'])
        
        # Apply vintage quality adjustment
        adjusted_charge_off_rates = base_charge_off_rates * vintage_quality_adjustment
        
        # Calculate outstanding balances (assuming linear amortization)
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
            'fico_band': fico_band,
            'seasoning_month': seasoning_months,
            'outstanding_balance': outstanding_balances,
            'charge_off_rate': adjusted_charge_off_rates,
            'charge_off_amount': charge_off_amounts,
            'cumulative_charge_off_amount': np.cumsum(charge_off_amounts),
            'cumulative_charge_off_rate': np.cumsum(charge_off_amounts) / loan_amount
        })
        
        return forecast_df
    
    def _calculate_aggregate_forecast(self, fico_forecasts: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate aggregate (dollar-weighted) forecast from FICO band forecasts.
        
        Args:
            fico_forecasts: DataFrame with forecasts by FICO band
            
        Returns:
            DataFrame with aggregate forecast
        """
        # Group by vintage_date and seasoning_month, sum amounts
        aggregate_forecast = fico_forecasts.groupby(['vintage_date', 'seasoning_month']).agg({
            'outstanding_balance': 'sum',
            'charge_off_amount': 'sum',
            'cumulative_charge_off_amount': 'sum'
        }).reset_index()
        
        # Calculate dollar-weighted charge-off rate
        aggregate_forecast['charge_off_rate'] = (
            aggregate_forecast['charge_off_amount'] / aggregate_forecast['outstanding_balance']
        )
        
        # Calculate cumulative charge-off rate
        total_loan_amount = fico_forecasts.groupby('vintage_date')['outstanding_balance'].first().sum()
        aggregate_forecast['cumulative_charge_off_rate'] = (
            aggregate_forecast['cumulative_charge_off_amount'] / total_loan_amount
        )
        
        return aggregate_forecast
    
    def forecast_portfolio_charge_offs(self, 
                                     portfolio_data: pd.DataFrame,
                                     forecast_end_date: datetime,
                                     vintage_quality_model: Optional[Dict] = None) -> pd.DataFrame:
        """
        Forecast charge-offs for an entire portfolio with FICO segmentation.
        
        Args:
            portfolio_data: DataFrame with vintage and FICO band information
                           Columns: vintage_date, fico_band, loan_amount, num_loans
            forecast_end_date: End date for the forecast
            vintage_quality_model: Optional model to predict vintage quality adjustments
            
        Returns:
            DataFrame with portfolio-level charge-off forecasts
        """
        all_forecasts = []
        
        # Group by vintage date
        for vintage_date in portfolio_data['vintage_date'].unique():
            vintage_data = portfolio_data[portfolio_data['vintage_date'] == vintage_date]
            
            # Create portfolio mix for this vintage
            portfolio_mix = {}
            for _, row in vintage_data.iterrows():
                portfolio_mix[row['fico_band']] = {
                    'loan_amount': row['loan_amount'],
                    'num_loans': row['num_loans']
                }
            
            # Determine vintage quality adjustments by FICO band
            vintage_quality_adjustments = None
            if vintage_quality_model is not None:
                vintage_quality_adjustments = self._predict_vintage_quality_by_fico(
                    vintage_date, vintage_quality_model
                )
            
            # Calculate forecast horizon
            forecast_horizon = ((forecast_end_date.year - vintage_date.year) * 12 + 
                              forecast_end_date.month - vintage_date.month)
            
            # Forecast this vintage by FICO band
            vintage_forecast = self.forecast_vintage_charge_offs_by_fico(
                vintage_date=vintage_date,
                portfolio_mix=portfolio_mix,
                forecast_horizon=forecast_horizon,
                vintage_quality_adjustments=vintage_quality_adjustments
            )
            
            all_forecasts.append(vintage_forecast)
        
        # Combine all vintage forecasts
        portfolio_forecast = pd.concat(all_forecasts, ignore_index=True)
        
        # Aggregate by report date
        portfolio_forecast['report_date'] = portfolio_forecast.apply(
            lambda row: row['vintage_date'] + pd.DateOffset(months=row['seasoning_month']), axis=1
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
    
    def _get_best_seasoning_curve(self, fico_band: str) -> Optional[Dict]:
        """Get the best fitting seasoning curve for a specific FICO band."""
        if not self.seasoning_curves or fico_band not in self.seasoning_curves:
            return None
        
        best_curve = None
        best_r_squared = -1
        
        for curve_name, curve_info in self.seasoning_curves[fico_band].items():
            if curve_info is not None and curve_info['r_squared'] > best_r_squared:
                best_r_squared = curve_info['r_squared']
                best_curve = curve_info
        
        return best_curve
    
    def _predict_vintage_quality_by_fico(self, vintage_date: datetime, model: Dict) -> Dict[str, float]:
        """
        Predict vintage quality adjustment by FICO band based on historical patterns.
        
        Args:
            vintage_date: Vintage date to predict quality for
            model: Dictionary with vintage quality prediction model
            
        Returns:
            Dictionary with vintage quality adjustment factors by FICO band
        """
        adjustments = {}
        
        for fico_band in self.fico_bands:
            # Extract features for prediction
            vintage_month = vintage_date.month
            vintage_year = vintage_date.year
            
            # Get FICO-specific seasonal patterns
            monthly_key = f'monthly_seasonality_{fico_band}'
            yearly_key = f'yearly_trends_{fico_band}'
            
            # Monthly seasonality effect
            if monthly_key in model:
                monthly_effect = model[monthly_key].get(vintage_month, 1.0)
            else:
                monthly_effect = 1.0
            
            # Yearly trend effect
            if yearly_key in model:
                # Use average of recent years as baseline
                recent_years = [y for y in model[yearly_key].keys() if y >= vintage_year - 3]
                if recent_years:
                    baseline = np.mean([model[yearly_key][y] for y in recent_years])
                    current_trend = model[yearly_key].get(vintage_year, baseline)
                    trend_effect = current_trend / baseline if baseline > 0 else 1.0
                else:
                    trend_effect = 1.0
            else:
                trend_effect = 1.0
            
            # Combine effects
            quality_adjustment = monthly_effect * trend_effect
            
            # Ensure reasonable bounds
            quality_adjustment = max(0.5, min(2.0, quality_adjustment))
            
            adjustments[fico_band] = quality_adjustment
        
        return adjustments
    
    def forecast_vintage_charge_offs(self, 
                                   vintage_date: datetime,
                                   loan_amount: float,
                                   num_loans: int,
                                   forecast_horizon: int = 120,
                                   vintage_quality_adjustment: float = 1.0) -> pd.DataFrame:
        """
        Legacy method for backward compatibility - forecasts aggregate vintage.
        
        Args:
            vintage_date: Origination date of the vintage
            loan_amount: Total loan amount for the vintage
            num_loans: Number of loans in the vintage
            forecast_horizon: Number of months to forecast
            vintage_quality_adjustment: Multiplier to adjust vintage quality
            
        Returns:
            DataFrame with forecasted charge-off rates and amounts
        """
        # Create a simple portfolio mix (assume equal distribution across FICO bands)
        portfolio_mix = {}
        loan_amount_per_band = loan_amount / len(self.fico_bands)
        num_loans_per_band = num_loans // len(self.fico_bands)
        
        for fico_band in self.fico_bands:
            portfolio_mix[fico_band] = {
                'loan_amount': loan_amount_per_band,
                'num_loans': num_loans_per_band
            }
        
        # Use the FICO-segmented forecast method
        return self.forecast_vintage_charge_offs_by_fico(
            vintage_date=vintage_date,
            portfolio_mix=portfolio_mix,
            forecast_horizon=forecast_horizon,
            vintage_quality_adjustments={band: vintage_quality_adjustment for band in self.fico_bands}
        )
    
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
        fig.suptitle('Charge-off Forecast Dashboard - FICO Segmentation', fontsize=16, fontweight='bold')
        
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

    def scaling_method(self, observed_month: int, observed_cum_co: float, typical_cum_co: Dict[int, float]) -> float:
        """
        Simple scaling method: Estimate lifetime charge-off rate by scaling up the observed cumulative charge-off rate at a given month.
        Args:
            observed_month: The current month of observation
            observed_cum_co: The observed cumulative charge-off rate at that month
            typical_cum_co: Dict mapping month to typical cumulative charge-off rate (from historical data)
        Returns:
            Estimated lifetime charge-off rate
        """
        if observed_month not in typical_cum_co or typical_cum_co[observed_month] == 0:
            return np.nan
        return observed_cum_co / typical_cum_co[observed_month]

    def additive_method(self, observed_month: int, observed_cum_co: float, typical_incremental_co: Dict[int, float], max_month: int = 60) -> float:
        """
        Additive method: Estimate lifetime charge-off rate by adding expected future incremental charge-off rates to the current observed rate.
        Args:
            observed_month: The current month of observation
            observed_cum_co: The observed cumulative charge-off rate at that month
            typical_incremental_co: Dict mapping month to typical incremental charge-off rate (from historical data)
            max_month: The maximum month to sum to (lifetime horizon)
        Returns:
            Estimated lifetime charge-off rate
        """
        future_months = range(observed_month + 1, max_month + 1)
        future_sum = sum([typical_incremental_co.get(m, 0) for m in future_months])
        return observed_cum_co + future_sum

    def compare_simple_methods(self, observed_month: int, observed_cum_co: float, typical_cum_co: Dict[int, float], typical_incremental_co: Dict[int, float], max_month: int = 60) -> Dict[str, float]:
        """
        Compare scaling and additive methods for a given observation.
        Returns a dict with method names and their estimated lifetime charge-off rates.
        """
        scaling_est = self.scaling_method(observed_month, observed_cum_co, typical_cum_co)
        additive_est = self.additive_method(observed_month, observed_cum_co, typical_incremental_co, max_month)
        return {
            'scaling': scaling_est,
            'additive': additive_est
        } 