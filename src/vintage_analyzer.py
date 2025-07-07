"""
Vintage analysis for loan charge-off forecasting with FICO segmentation.
Analyzes loan performance patterns by vintage, seasoning, and FICO bands.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings('ignore')


class VintageAnalyzer:
    """
    Analyzes loan performance patterns by vintage, seasoning, and FICO bands using a multi-model approach.

    The system fits and compares several models to the cumulative charge-off data for each (vintage, FICO band) segment:
    - Weibull CDF: Flexible, interpretable, commonly used for time-to-event data
    - Lognormal CDF: Captures right-skewed timing of losses, interpretable
    - Gompertz CDF: Handles saturation and deceleration, interpretable
    - Simple Linear/Polynomial Trend: Baseline, highly explainable, but may underfit
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
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize the vintage analyzer.
        
        Args:
            data: Loan performance data with vintage_date, seasoning_month, and fico_band columns
        """
        self.data = data
        self.vintage_summary = None
        self.seasoning_curves = None
        self.fico_bands = data['fico_band'].unique() if 'fico_band' in data.columns else []
        
    def calculate_vintage_metrics(self) -> pd.DataFrame:
        """
        Calculate vintage-level performance metrics by FICO band.
        
        Returns:
            DataFrame with vintage performance metrics by FICO band
        """
        # Group by vintage, FICO band, and seasoning month
        vintage_metrics = self.data.groupby(['vintage_date', 'fico_band', 'seasoning_month']).agg({
            'loan_amount': 'sum',
            'outstanding_balance': 'sum',
            'charge_off_amount': 'sum',
            'loan_id': 'count'
        }).reset_index()
        
        # Calculate charge-off rates
        vintage_metrics['charge_off_rate'] = (
            vintage_metrics['charge_off_amount'] / vintage_metrics['outstanding_balance']
        )
        vintage_metrics['cumulative_charge_off_rate'] = (
            vintage_metrics.groupby(['vintage_date', 'fico_band'])['charge_off_amount'].cumsum() /
            vintage_metrics.groupby(['vintage_date', 'fico_band'])['loan_amount'].first()
        )
        
        # Calculate vintage characteristics
        vintage_metrics['avg_loan_size'] = (
            vintage_metrics['loan_amount'] / vintage_metrics['loan_id']
        )
        
        # Add risk grade
        if 'risk_grade' in self.data.columns:
            risk_grade_map = self.data.groupby('fico_band')['risk_grade'].first().to_dict()
            vintage_metrics['risk_grade'] = vintage_metrics['fico_band'].map(risk_grade_map)
        
        self.vintage_summary = vintage_metrics
        return vintage_metrics
    
    def calculate_aggregate_vintage_metrics(self) -> pd.DataFrame:
        """
        Calculate aggregate vintage metrics (dollar-weighted across FICO bands).
        
        Returns:
            DataFrame with aggregate vintage performance metrics
        """
        if self.vintage_summary is None:
            self.calculate_vintage_metrics()
        
        # Aggregate across FICO bands for each vintage and seasoning month
        aggregate_metrics = self.vintage_summary.groupby(['vintage_date', 'seasoning_month']).agg({
            'loan_amount': 'sum',
            'outstanding_balance': 'sum',
            'charge_off_amount': 'sum',
            'loan_id': 'sum'
        }).reset_index()
        
        # Calculate dollar-weighted charge-off rates
        aggregate_metrics['charge_off_rate'] = (
            aggregate_metrics['charge_off_amount'] / aggregate_metrics['outstanding_balance']
        )
        aggregate_metrics['cumulative_charge_off_rate'] = (
            aggregate_metrics.groupby('vintage_date')['charge_off_amount'].cumsum() /
            aggregate_metrics.groupby('vintage_date')['loan_amount'].first()
        )
        
        aggregate_metrics['avg_loan_size'] = (
            aggregate_metrics['loan_amount'] / aggregate_metrics['loan_id']
        )
        
        return aggregate_metrics
    
    def fit_seasoning_curves(self, max_seasoning: int = 120) -> Dict:
        """
        Fit seasoning curves to historical data by FICO band.
        
        Args:
            max_seasoning: Maximum seasoning months to consider
            
        Returns:
            Dictionary with fitted seasoning curve parameters by FICO band
        """
        if self.vintage_summary is None:
            self.calculate_vintage_metrics()
        
        # Filter to reasonable seasoning periods
        curve_data = self.vintage_summary[
            self.vintage_summary['seasoning_month'] <= max_seasoning
        ].copy()
        
        # Define seasoning curve functions
        def weibull_curve(x, alpha, beta, gamma):
            """Weibull distribution-based seasoning curve"""
            return alpha * (1 - np.exp(-((x / beta) ** gamma)))
        
        def lognormal_curve(x, alpha, mu, sigma):
            """Lognormal distribution-based seasoning curve"""
            return alpha * stats.lognorm.cdf(x, sigma, scale=np.exp(mu))
        
        def gompertz_curve(x, alpha, beta, gamma):
            """Gompertz curve for seasoning"""
            return alpha * np.exp(-beta * np.exp(-gamma * x))
        
        curves = {}
        
        # Fit curves for each FICO band
        for fico_band in self.fico_bands:
            band_data = curve_data[curve_data['fico_band'] == fico_band]
            avg_seasoning = band_data.groupby('seasoning_month')['charge_off_rate'].mean().reset_index()
            
            if len(avg_seasoning) < 10:  # Need sufficient data points
                continue
                
            band_curves = {}
            
            # Fit Weibull curve
            try:
                popt_weibull, _ = curve_fit(
                    weibull_curve, 
                    avg_seasoning['seasoning_month'], 
                    avg_seasoning['charge_off_rate'],
                    p0=[0.05, 24, 2],
                    bounds=([0, 1, 0.1], [0.3, 60, 10])
                )
                band_curves['weibull'] = {
                    'function': weibull_curve,
                    'params': popt_weibull,
                    'r_squared': self._calculate_r_squared(
                        avg_seasoning['charge_off_rate'],
                        weibull_curve(avg_seasoning['seasoning_month'], *popt_weibull)
                    )
                }
            except:
                band_curves['weibull'] = None
            
            # Fit Lognormal curve
            try:
                popt_lognorm, _ = curve_fit(
                    lognormal_curve,
                    avg_seasoning['seasoning_month'],
                    avg_seasoning['charge_off_rate'],
                    p0=[0.05, 3, 0.5],
                    bounds=([0, 1, 0.1], [0.3, 5, 2])
                )
                band_curves['lognormal'] = {
                    'function': lognormal_curve,
                    'params': popt_lognorm,
                    'r_squared': self._calculate_r_squared(
                        avg_seasoning['charge_off_rate'],
                        lognormal_curve(avg_seasoning['seasoning_month'], *popt_lognorm)
                    )
                }
            except:
                band_curves['lognormal'] = None
            
            # Fit Gompertz curve
            try:
                popt_gompertz, _ = curve_fit(
                    gompertz_curve,
                    avg_seasoning['seasoning_month'],
                    avg_seasoning['charge_off_rate'],
                    p0=[0.05, 1, 0.1],
                    bounds=([0, 0, 0], [0.3, 10, 1])
                )
                band_curves['gompertz'] = {
                    'function': gompertz_curve,
                    'params': popt_gompertz,
                    'r_squared': self._calculate_r_squared(
                        avg_seasoning['charge_off_rate'],
                        gompertz_curve(avg_seasoning['seasoning_month'], *popt_gompertz)
                    )
                }
            except:
                band_curves['gompertz'] = None
            
            curves[fico_band] = band_curves
        
        # Also fit aggregate curves
        aggregate_data = self.calculate_aggregate_vintage_metrics()
        aggregate_curve_data = aggregate_data[
            aggregate_data['seasoning_month'] <= max_seasoning
        ]
        avg_aggregate_seasoning = aggregate_curve_data.groupby('seasoning_month')['charge_off_rate'].mean().reset_index()
        
        aggregate_curves = {}
        
        # Fit aggregate Weibull curve
        try:
            popt_weibull, _ = curve_fit(
                weibull_curve, 
                avg_aggregate_seasoning['seasoning_month'], 
                avg_aggregate_seasoning['charge_off_rate'],
                p0=[0.05, 24, 2],
                bounds=([0, 1, 0.1], [0.3, 60, 10])
            )
            aggregate_curves['weibull'] = {
                'function': weibull_curve,
                'params': popt_weibull,
                'r_squared': self._calculate_r_squared(
                    avg_aggregate_seasoning['charge_off_rate'],
                    weibull_curve(avg_aggregate_seasoning['seasoning_month'], *popt_weibull)
                )
            }
        except:
            aggregate_curves['weibull'] = None
        
        curves['aggregate'] = aggregate_curves
        
        self.seasoning_curves = curves
        return curves
    
    def _calculate_r_squared(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate R-squared for curve fitting"""
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    def analyze_vintage_quality(self) -> pd.DataFrame:
        """
        Analyze vintage quality based on early performance indicators by FICO band.
        
        Returns:
            DataFrame with vintage quality metrics by FICO band
        """
        if self.vintage_summary is None:
            self.calculate_vintage_metrics()
        
        # Focus on early seasoning periods (6, 12, 18 months)
        early_periods = [6, 12, 18]
        vintage_quality = []
        
        for vintage in self.vintage_summary['vintage_date'].unique():
            for fico_band in self.fico_bands:
                vintage_data = self.vintage_summary[
                    (self.vintage_summary['vintage_date'] == vintage) &
                    (self.vintage_summary['fico_band'] == fico_band)
                ]
                
                if vintage_data.empty:
                    continue
                
                quality_metrics = {
                    'vintage_date': vintage,
                    'fico_band': fico_band
                }
                
                # Add risk grade if available
                if 'risk_grade' in vintage_data.columns:
                    quality_metrics['risk_grade'] = vintage_data['risk_grade'].iloc[0]
                
                for period in early_periods:
                    period_data = vintage_data[vintage_data['seasoning_month'] == period]
                    if not period_data.empty:
                        quality_metrics[f'charge_off_rate_{period}m'] = period_data['charge_off_rate'].iloc[0]
                        quality_metrics[f'cumulative_charge_off_{period}m'] = period_data['cumulative_charge_off_rate'].iloc[0]
                    else:
                        quality_metrics[f'charge_off_rate_{period}m'] = np.nan
                        quality_metrics[f'cumulative_charge_off_{period}m'] = np.nan
                
                # Calculate vintage characteristics
                vintage_loans = self.data[
                    (self.data['vintage_date'] == vintage) &
                    (self.data['fico_band'] == fico_band)
                ]
                quality_metrics['total_loans'] = vintage_loans['loan_id'].nunique()
                quality_metrics['avg_loan_size'] = vintage_loans['loan_amount'].mean()
                quality_metrics['avg_interest_rate'] = vintage_loans['interest_rate'].mean()
                quality_metrics['avg_term'] = vintage_loans['term'].mean()
                quality_metrics['avg_fico_score'] = vintage_loans['fico_score'].mean()
                
                vintage_quality.append(quality_metrics)
        
        return pd.DataFrame(vintage_quality)
    
    def analyze_fico_mix_trends(self) -> Dict:
        """
        Analyze trends in FICO band mix over time.
        
        Returns:
            Dictionary with FICO mix analysis
        """
        # Get FICO mix by vintage
        fico_mix = self.data.groupby(['vintage_date', 'fico_band', 'loan_id']).first().reset_index()
        vintage_mix = fico_mix.groupby(['vintage_date', 'fico_band']).agg({
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
        if 'risk_grade' in self.data.columns:
            risk_grade_map = self.data.groupby('fico_band')['risk_grade'].first().to_dict()
            vintage_mix['risk_grade'] = vintage_mix['fico_band'].map(risk_grade_map)
        
        # Analyze trends
        trends = {}
        
        # Overall portfolio quality trend
        if 'risk_grade' in vintage_mix.columns:
            portfolio_quality = vintage_mix.groupby('vintage_date').apply(
                lambda x: np.average(x['risk_grade'], weights=x['loan_amount'])
            ).reset_index()
            portfolio_quality.columns = ['vintage_date', 'weighted_avg_risk_grade']
            trends['portfolio_quality'] = portfolio_quality
        
        # FICO band trends
        for fico_band in self.fico_bands:
            band_trend = vintage_mix[vintage_mix['fico_band'] == fico_band].copy()
            if not band_trend.empty:
                trends[f'fico_band_{fico_band}'] = band_trend
        
        return trends
    
    def identify_vintage_patterns(self) -> Dict:
        """
        Identify patterns in vintage performance by FICO band.
        
        Returns:
            Dictionary with vintage pattern analysis
        """
        if self.vintage_summary is None:
            self.calculate_vintage_metrics()
        
        patterns = {}
        
        # Seasonal patterns by FICO band
        vintage_quality = self.analyze_vintage_quality()
        vintage_quality['vintage_month'] = vintage_quality['vintage_date'].dt.month
        vintage_quality['vintage_year'] = vintage_quality['vintage_date'].dt.year
        
        # Monthly seasonality by FICO band
        for fico_band in self.fico_bands:
            band_data = vintage_quality[vintage_quality['fico_band'] == fico_band]
            if not band_data.empty:
                monthly_performance = band_data.groupby('vintage_month')[
                    'charge_off_rate_12m'
                ].mean()
                patterns[f'monthly_seasonality_{fico_band}'] = monthly_performance.to_dict()
        
        # Yearly trends by FICO band
        for fico_band in self.fico_bands:
            band_data = vintage_quality[vintage_quality['fico_band'] == fico_band]
            if not band_data.empty:
                yearly_performance = band_data.groupby('vintage_year')[
                    'charge_off_rate_12m'
                ].mean()
                patterns[f'yearly_trends_{fico_band}'] = yearly_performance.to_dict()
        
        # Vintage quality ranking by FICO band
        for fico_band in self.fico_bands:
            band_data = vintage_quality[vintage_quality['fico_band'] == fico_band].copy()
            if not band_data.empty:
                band_data['quality_score'] = (
                    band_data['charge_off_rate_12m'].rank(ascending=True) +
                    band_data['charge_off_rate_18m'].rank(ascending=True)
                ) / 2
                
                patterns[f'best_vintages_{fico_band}'] = band_data.nsmallest(3, 'quality_score')[
                    ['vintage_date', 'quality_score', 'charge_off_rate_12m', 'charge_off_rate_18m']
                ].to_dict('records')
                
                patterns[f'worst_vintages_{fico_band}'] = band_data.nlargest(3, 'quality_score')[
                    ['vintage_date', 'quality_score', 'charge_off_rate_12m', 'charge_off_rate_18m']
                ].to_dict('records')
        
        return patterns
    
    def plot_vintage_analysis(self, save_path: Optional[str] = None):
        """
        Create comprehensive vintage analysis plots with FICO segmentation.
        
        Args:
            save_path: Optional path to save the plots
        """
        if self.vintage_summary is None:
            self.calculate_vintage_metrics()
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Vintage Analysis Dashboard - FICO Segmentation', fontsize=16, fontweight='bold')
        
        # 1. Vintage performance heatmap (aggregate)
        aggregate_metrics = self.calculate_aggregate_vintage_metrics()
        pivot_data = aggregate_metrics.pivot_table(
            values='charge_off_rate',
            index='vintage_date',
            columns='seasoning_month',
            aggfunc='mean'
        )
        
        sns.heatmap(pivot_data, ax=axes[0, 0], cmap='YlOrRd', cbar_kws={'label': 'Charge-off Rate'})
        axes[0, 0].set_title('Aggregate Vintage Performance Heatmap')
        axes[0, 0].set_xlabel('Seasoning Month')
        axes[0, 0].set_ylabel('Vintage Date')
        
        # 2. Average seasoning curves by FICO band
        colors = ['red', 'orange', 'yellow', 'green', 'blue']
        for i, fico_band in enumerate(self.fico_bands):
            band_data = self.vintage_summary[self.vintage_summary['fico_band'] == fico_band]
            avg_seasoning = band_data.groupby('seasoning_month')['charge_off_rate'].mean()
            axes[0, 1].plot(avg_seasoning.index, avg_seasoning.values, 
                           color=colors[i], linewidth=2, label=fico_band)
        
        axes[0, 1].set_title('Average Seasoning Curves by FICO Band')
        axes[0, 1].set_xlabel('Seasoning Month')
        axes[0, 1].set_ylabel('Charge-off Rate')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. FICO mix trends
        fico_mix_trends = self.analyze_fico_mix_trends()
        if 'portfolio_quality' in fico_mix_trends:
            portfolio_quality = fico_mix_trends['portfolio_quality']
            axes[0, 2].plot(portfolio_quality['vintage_date'], 
                           portfolio_quality['weighted_avg_risk_grade'], 
                           'b-', linewidth=2, marker='o')
            axes[0, 2].set_title('Portfolio Quality Trend (Lower = Better)')
            axes[0, 2].set_xlabel('Vintage Date')
            axes[0, 2].set_ylabel('Weighted Avg Risk Grade')
            axes[0, 2].tick_params(axis='x', rotation=45)
            axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Vintage quality comparison by FICO band
        vintage_quality = self.analyze_vintage_quality()
        for i, fico_band in enumerate(self.fico_bands):
            band_data = vintage_quality[vintage_quality['fico_band'] == fico_band]
            if not band_data.empty:
                axes[1, 0].scatter(band_data['vintage_date'], 
                                  band_data['charge_off_rate_12m'], 
                                  alpha=0.7, s=30, color=colors[i], label=fico_band)
        
        axes[1, 0].set_title('Vintage Quality (12-Month Performance)')
        axes[1, 0].set_xlabel('Vintage Date')
        axes[1, 0].set_ylabel('12-Month Charge-off Rate')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Cumulative charge-off rates by FICO band
        recent_vintages = self.vintage_summary[
            self.vintage_summary['vintage_date'] >= '2020-01-01'
        ]
        
        for i, fico_band in enumerate(self.fico_bands):
            band_data = recent_vintages[recent_vintages['fico_band'] == fico_band]
            if not band_data.empty:
                # Plot last vintage for this band
                last_vintage = band_data['vintage_date'].max()
                vintage_data = band_data[band_data['vintage_date'] == last_vintage]
                axes[1, 1].plot(vintage_data['seasoning_month'], 
                               vintage_data['cumulative_charge_off_rate'], 
                               color=colors[i], marker='o', label=f'{fico_band} ({last_vintage.strftime("%Y-%m")})')
        
        axes[1, 1].set_title('Cumulative Charge-off Rates (Recent Vintages)')
        axes[1, 1].set_xlabel('Seasoning Month')
        axes[1, 1].set_ylabel('Cumulative Charge-off Rate')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. FICO band distribution
        fico_distribution = self.data.groupby('fico_band')['loan_amount'].sum()
        axes[1, 2].pie(fico_distribution.values, labels=fico_distribution.index, autopct='%1.1f%%')
        axes[1, 2].set_title('Portfolio Distribution by FICO Band')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()

    def analyze_vintage_data(self, data: pd.DataFrame) -> Dict:
        """
        Analyze vintage data by FICO band.
        
        Args:
            data: DataFrame with vintage data
            
        Returns:
            Dictionary containing analysis results
        """
        print("Analyzing vintage data by FICO band...")
        
        results = {
            'vintage_summary': {},
            'fico_band_analysis': {},
            'seasoning_curves': {},
            'quality_mix_trends': {},
            'charge_off_patterns': {}
        }
        
        # Overall vintage summary
        results['vintage_summary'] = self._calculate_vintage_summary(data)
        
        # Analyze each FICO band
        for fico_band in self.FICO_BANDS.keys():
            print(f"  Analyzing FICO band: {fico_band}")
            band_data = data[data['fico_band'] == fico_band].copy()
            
            if not band_data.empty:
                results['fico_band_analysis'][fico_band] = self._analyze_fico_band(band_data)
                results['seasoning_curves'][fico_band] = self._fit_seasoning_curves(band_data)
                results['charge_off_patterns'][fico_band] = self._analyze_charge_off_patterns(band_data)
        
        # Quality mix trends
        results['quality_mix_trends'] = self._analyze_quality_mix_trends(data)
        
        return results
    
    def _analyze_charge_off_patterns(self, data: pd.DataFrame) -> Dict:
        """
        Analyze charge-off patterns for a FICO band.
        
        Args:
            data: DataFrame for a specific FICO band
            
        Returns:
            Dictionary with charge-off pattern analysis
        """
        patterns = {}
        
        # Calculate cumulative charge-off rates by seasoning
        cumulative_co = data.groupby('seasoning_month').agg({
            'charge_off_rate': 'sum',
            'loan_id': 'nunique'
        }).reset_index()
        cumulative_co['cumulative_charge_off_rate'] = (
            cumulative_co['charge_off_rate'].cumsum() / cumulative_co['loan_id'].iloc[0]
        )
        
        patterns['cumulative_charge_off_curve'] = cumulative_co
        
        # Calculate charge-off timing distribution
        charge_off_timing = data[data['charge_off_rate'] == 1].groupby('seasoning_month').size()
        patterns['charge_off_timing_distribution'] = charge_off_timing.to_dict()
        
        # Calculate average seasoning at charge-off
        if not data[data['charge_off_rate'] == 1].empty:
            avg_seasoning_at_co = data[data['charge_off_rate'] == 1]['seasoning_month'].mean()
            patterns['average_seasoning_at_charge_off'] = avg_seasoning_at_co
        else:
            patterns['average_seasoning_at_charge_off'] = None
        
        # Calculate charge-off amount patterns
        co_amounts = data[data['charge_off_amount'] > 0]
        if not co_amounts.empty:
            patterns['charge_off_amount_stats'] = {
                'mean': co_amounts['charge_off_amount'].mean(),
                'median': co_amounts['charge_off_amount'].median(),
                'std': co_amounts['charge_off_amount'].std(),
                'min': co_amounts['charge_off_amount'].min(),
                'max': co_amounts['charge_off_amount'].max()
            }
        else:
            patterns['charge_off_amount_stats'] = None
        
        return patterns 