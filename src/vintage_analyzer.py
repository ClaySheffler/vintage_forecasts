"""
Vintage analysis for loan charge-off forecasting.
Analyzes loan performance patterns by vintage and seasoning periods.
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
    Analyzes loan performance patterns by vintage and seasoning.
    """
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize the vintage analyzer.
        
        Args:
            data: Loan performance data with vintage_date and seasoning_month columns
        """
        self.data = data
        self.vintage_summary = None
        self.seasoning_curves = None
        
    def calculate_vintage_metrics(self) -> pd.DataFrame:
        """
        Calculate vintage-level performance metrics.
        
        Returns:
            DataFrame with vintage performance metrics
        """
        # Group by vintage and seasoning month
        vintage_metrics = self.data.groupby(['vintage_date', 'seasoning_month']).agg({
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
            vintage_metrics.groupby('vintage_date')['charge_off_amount'].cumsum() /
            vintage_metrics.groupby('vintage_date')['loan_amount'].first()
        )
        
        # Calculate vintage characteristics
        vintage_metrics['avg_loan_size'] = (
            vintage_metrics['loan_amount'] / vintage_metrics['loan_id']
        )
        
        self.vintage_summary = vintage_metrics
        return vintage_metrics
    
    def fit_seasoning_curves(self, max_seasoning: int = 120) -> Dict:
        """
        Fit seasoning curves to historical data.
        
        Args:
            max_seasoning: Maximum seasoning months to consider
            
        Returns:
            Dictionary with fitted seasoning curve parameters
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
        
        # Fit curves to average seasoning pattern
        avg_seasoning = curve_data.groupby('seasoning_month')['charge_off_rate'].mean().reset_index()
        
        curves = {}
        
        # Fit Weibull curve
        try:
            popt_weibull, _ = curve_fit(
                weibull_curve, 
                avg_seasoning['seasoning_month'], 
                avg_seasoning['charge_off_rate'],
                p0=[0.05, 24, 2],  # Initial guesses
                bounds=([0, 1, 0.1], [0.2, 60, 10])
            )
            curves['weibull'] = {
                'function': weibull_curve,
                'params': popt_weibull,
                'r_squared': self._calculate_r_squared(
                    avg_seasoning['charge_off_rate'],
                    weibull_curve(avg_seasoning['seasoning_month'], *popt_weibull)
                )
            }
        except:
            curves['weibull'] = None
        
        # Fit Lognormal curve
        try:
            popt_lognorm, _ = curve_fit(
                lognormal_curve,
                avg_seasoning['seasoning_month'],
                avg_seasoning['charge_off_rate'],
                p0=[0.05, 3, 0.5],
                bounds=([0, 1, 0.1], [0.2, 5, 2])
            )
            curves['lognormal'] = {
                'function': lognormal_curve,
                'params': popt_lognorm,
                'r_squared': self._calculate_r_squared(
                    avg_seasoning['charge_off_rate'],
                    lognormal_curve(avg_seasoning['seasoning_month'], *popt_lognorm)
                )
            }
        except:
            curves['lognormal'] = None
        
        # Fit Gompertz curve
        try:
            popt_gompertz, _ = curve_fit(
                gompertz_curve,
                avg_seasoning['seasoning_month'],
                avg_seasoning['charge_off_rate'],
                p0=[0.05, 1, 0.1],
                bounds=([0, 0, 0], [0.2, 10, 1])
            )
            curves['gompertz'] = {
                'function': gompertz_curve,
                'params': popt_gompertz,
                'r_squared': self._calculate_r_squared(
                    avg_seasoning['charge_off_rate'],
                    gompertz_curve(avg_seasoning['seasoning_month'], *popt_gompertz)
                )
            }
        except:
            curves['gompertz'] = None
        
        self.seasoning_curves = curves
        return curves
    
    def _calculate_r_squared(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate R-squared for curve fitting"""
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    def analyze_vintage_quality(self) -> pd.DataFrame:
        """
        Analyze vintage quality based on early performance indicators.
        
        Returns:
            DataFrame with vintage quality metrics
        """
        if self.vintage_summary is None:
            self.calculate_vintage_metrics()
        
        # Focus on early seasoning periods (6, 12, 18 months)
        early_periods = [6, 12, 18]
        vintage_quality = []
        
        for vintage in self.vintage_summary['vintage_date'].unique():
            vintage_data = self.vintage_summary[
                self.vintage_summary['vintage_date'] == vintage
            ]
            
            quality_metrics = {'vintage_date': vintage}
            
            for period in early_periods:
                period_data = vintage_data[vintage_data['seasoning_month'] == period]
                if not period_data.empty:
                    quality_metrics[f'charge_off_rate_{period}m'] = period_data['charge_off_rate'].iloc[0]
                    quality_metrics[f'cumulative_charge_off_{period}m'] = period_data['cumulative_charge_off_rate'].iloc[0]
                else:
                    quality_metrics[f'charge_off_rate_{period}m'] = np.nan
                    quality_metrics[f'cumulative_charge_off_{period}m'] = np.nan
            
            # Calculate vintage characteristics
            vintage_loans = self.data[self.data['vintage_date'] == vintage]
            quality_metrics['total_loans'] = vintage_loans['loan_id'].nunique()
            quality_metrics['avg_loan_size'] = vintage_loans['loan_amount'].mean()
            quality_metrics['avg_interest_rate'] = vintage_loans['interest_rate'].mean()
            quality_metrics['avg_term'] = vintage_loans['term'].mean()
            
            vintage_quality.append(quality_metrics)
        
        return pd.DataFrame(vintage_quality)
    
    def identify_vintage_patterns(self) -> Dict:
        """
        Identify patterns in vintage performance.
        
        Returns:
            Dictionary with vintage pattern analysis
        """
        if self.vintage_summary is None:
            self.calculate_vintage_metrics()
        
        patterns = {}
        
        # Seasonal patterns
        vintage_quality = self.analyze_vintage_quality()
        vintage_quality['vintage_month'] = vintage_quality['vintage_date'].dt.month
        vintage_quality['vintage_year'] = vintage_quality['vintage_date'].dt.year
        
        # Monthly seasonality
        monthly_performance = vintage_quality.groupby('vintage_month')[
            'charge_off_rate_12m'
        ].mean()
        
        patterns['monthly_seasonality'] = monthly_performance.to_dict()
        
        # Yearly trends
        yearly_performance = vintage_quality.groupby('vintage_year')[
            'charge_off_rate_12m'
        ].mean()
        
        patterns['yearly_trends'] = yearly_performance.to_dict()
        
        # Vintage quality ranking
        vintage_quality['quality_score'] = (
            vintage_quality['charge_off_rate_12m'].rank(ascending=True) +
            vintage_quality['charge_off_rate_18m'].rank(ascending=True)
        ) / 2
        
        patterns['best_vintages'] = vintage_quality.nsmallest(5, 'quality_score')[
            ['vintage_date', 'quality_score', 'charge_off_rate_12m', 'charge_off_rate_18m']
        ].to_dict('records')
        
        patterns['worst_vintages'] = vintage_quality.nlargest(5, 'quality_score')[
            ['vintage_date', 'quality_score', 'charge_off_rate_12m', 'charge_off_rate_18m']
        ].to_dict('records')
        
        return patterns
    
    def plot_vintage_analysis(self, save_path: Optional[str] = None):
        """
        Create comprehensive vintage analysis plots.
        
        Args:
            save_path: Optional path to save the plots
        """
        if self.vintage_summary is None:
            self.calculate_vintage_metrics()
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Vintage Analysis Dashboard', fontsize=16, fontweight='bold')
        
        # 1. Vintage performance heatmap
        pivot_data = self.vintage_summary.pivot_table(
            values='charge_off_rate',
            index='vintage_date',
            columns='seasoning_month',
            aggfunc='mean'
        )
        
        sns.heatmap(pivot_data, ax=axes[0, 0], cmap='YlOrRd', cbar_kws={'label': 'Charge-off Rate'})
        axes[0, 0].set_title('Vintage Performance Heatmap')
        axes[0, 0].set_xlabel('Seasoning Month')
        axes[0, 0].set_ylabel('Vintage Date')
        
        # 2. Average seasoning curve
        avg_seasoning = self.vintage_summary.groupby('seasoning_month')['charge_off_rate'].mean()
        axes[0, 1].plot(avg_seasoning.index, avg_seasoning.values, 'b-', linewidth=2, label='Average')
        
        # Add fitted curves if available
        if self.seasoning_curves:
            x_range = np.arange(0, max(avg_seasoning.index) + 1)
            for curve_name, curve_info in self.seasoning_curves.items():
                if curve_info is not None:
                    y_pred = curve_info['function'](x_range, *curve_info['params'])
                    axes[0, 1].plot(x_range, y_pred, '--', alpha=0.7, 
                                   label=f'{curve_name} (R²={curve_info["r_squared"]:.3f})')
        
        axes[0, 1].set_title('Average Seasoning Curve')
        axes[0, 1].set_xlabel('Seasoning Month')
        axes[0, 1].set_ylabel('Charge-off Rate')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Vintage quality comparison
        vintage_quality = self.analyze_vintage_quality()
        axes[1, 0].scatter(vintage_quality['vintage_date'], 
                          vintage_quality['charge_off_rate_12m'], 
                          alpha=0.7, s=50)
        axes[1, 0].set_title('Vintage Quality (12-Month Performance)')
        axes[1, 0].set_xlabel('Vintage Date')
        axes[1, 0].set_ylabel('12-Month Charge-off Rate')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Cumulative charge-off rates
        recent_vintages = self.vintage_summary[
            self.vintage_summary['vintage_date'] >= '2020-01-01'
        ]
        
        for vintage in recent_vintages['vintage_date'].unique()[-5:]:  # Last 5 vintages
            vintage_data = recent_vintages[recent_vintages['vintage_date'] == vintage]
            axes[1, 1].plot(vintage_data['seasoning_month'], 
                           vintage_data['cumulative_charge_off_rate'], 
                           marker='o', label=vintage.strftime('%Y-%m'))
        
        axes[1, 1].set_title('Cumulative Charge-off Rates (Recent Vintages)')
        axes[1, 1].set_xlabel('Seasoning Month')
        axes[1, 1].set_ylabel('Cumulative Charge-off Rate')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show() 