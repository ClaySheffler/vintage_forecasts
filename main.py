"""
Main script for vintage charge-off forecasting.
Demonstrates the complete workflow from data loading to forecasting.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from src.data_loader import LoanDataLoader
from src.vintage_analyzer import VintageAnalyzer
from src.forecaster import ChargeOffForecaster


def main():
    """
    Main function demonstrating the vintage charge-off forecasting workflow.
    """
    print("=== Vintage Charge-off Forecasting System ===\n")
    
    # Step 1: Load and preprocess data
    print("1. Loading and preprocessing loan performance data...")
    data_loader = LoanDataLoader()
    loan_data = data_loader.load_sample_data()
    loan_data = data_loader.preprocess_data()
    
    print(f"   Loaded {len(loan_data):,} loan performance records")
    print(f"   Date range: {loan_data['vintage_date'].min()} to {loan_data['vintage_date'].max()}")
    print(f"   Total loans: {loan_data['loan_id'].nunique():,}")
    print(f"   Total loan amount: ${loan_data['loan_amount'].sum():,.0f}\n")
    
    # Step 2: Perform vintage analysis
    print("2. Performing vintage analysis...")
    vintage_analyzer = VintageAnalyzer(loan_data)
    vintage_metrics = vintage_analyzer.calculate_vintage_metrics()
    seasoning_curves = vintage_analyzer.fit_seasoning_curves()
    
    print("   Fitted seasoning curves:")
    for curve_name, curve_info in seasoning_curves.items():
        if curve_info is not None:
            print(f"     {curve_name}: R² = {curve_info['r_squared']:.3f}")
    
    # Analyze vintage patterns
    vintage_patterns = vintage_analyzer.identify_vintage_patterns()
    print(f"   Identified {len(vintage_patterns['best_vintages'])} best performing vintages")
    print(f"   Identified {len(vintage_patterns['worst_vintages'])} worst performing vintages\n")
    
    # Step 3: Create forecaster
    print("3. Initializing charge-off forecaster...")
    forecaster = ChargeOffForecaster(vintage_analyzer, loan_data)
    
    # Step 4: Generate portfolio forecast
    print("4. Generating portfolio charge-off forecast...")
    
    # Create sample portfolio data for forecasting
    portfolio_data = []
    current_date = datetime(2024, 1, 1)
    
    # Generate monthly vintages for the next 12 months
    for i in range(12):
        vintage_date = current_date + timedelta(days=30*i)
        loan_amount = np.random.uniform(50_000_000, 100_000_000)  # $50M-$100M per month
        num_loans = np.random.randint(2000, 5000)
        
        portfolio_data.append({
            'vintage_date': vintage_date,
            'loan_amount': loan_amount,
            'num_loans': num_loans
        })
    
    portfolio_df = pd.DataFrame(portfolio_data)
    forecast_end_date = datetime(2034, 12, 31)  # 10-year forecast horizon
    
    # Generate base forecast
    base_forecast = forecaster.forecast_portfolio_charge_offs(
        portfolio_data=portfolio_df,
        forecast_end_date=forecast_end_date,
        vintage_quality_model=vintage_patterns
    )
    
    print(f"   Forecast period: {base_forecast['report_date'].min()} to {base_forecast['report_date'].max()}")
    print(f"   Total forecasted charge-offs: ${base_forecast['charge_off_amount'].sum():,.0f}")
    print(f"   Peak charge-off rate: {base_forecast['charge_off_rate'].max():.2%}")
    print(f"   Average charge-off rate: {base_forecast['charge_off_rate'].mean():.2%}\n")
    
    # Step 5: Generate scenario forecasts
    print("5. Generating scenario forecasts...")
    scenarios = {
        'Optimistic': 0.7,    # 30% reduction in charge-offs
        'Base Case': 1.0,     # Base forecast
        'Pessimistic': 1.5,   # 50% increase in charge-offs
        'Severe Stress': 2.0   # 100% increase in charge-offs
    }
    
    scenario_forecasts = forecaster.generate_scenario_forecasts(base_forecast, scenarios)
    
    print("   Scenario forecasts generated:")
    for scenario_name, scenario_df in scenario_forecasts.items():
        total_charge_offs = scenario_df['charge_off_amount'].sum()
        peak_rate = scenario_df['charge_off_rate'].max()
        print(f"     {scenario_name}: ${total_charge_offs:,.0f} total, {peak_rate:.2%} peak rate")
    
    print()
    
    # Step 6: Calculate and display key metrics
    print("6. Calculating forecast metrics...")
    metrics = forecaster.calculate_forecast_metrics(base_forecast)
    
    print("   Key Forecast Metrics:")
    print(f"     Total Charge-offs: ${metrics['total_charge_offs']:,.0f}")
    print(f"     Peak Charge-off Rate: {metrics['peak_charge_off_rate']:.2%}")
    print(f"     Peak Month: {metrics['peak_charge_off_month'].strftime('%Y-%m')}")
    print(f"     Average Charge-off Rate: {metrics['avg_charge_off_rate']:.2%}")
    
    if 'median_charge_off_timing' in metrics:
        print(f"     Median Charge-off Timing: {metrics['median_charge_off_timing'].strftime('%Y-%m')}")
    
    print()
    
    # Step 7: Create visualizations
    print("7. Creating visualizations...")
    
    # Vintage analysis plots
    vintage_analyzer.plot_vintage_analysis(save_path='outputs/vintage_analysis.png')
    
    # Forecast plots
    forecaster.plot_forecast_results(
        base_forecast, 
        scenario_forecasts, 
        save_path='outputs/forecast_results.png'
    )
    
    print("   Plots saved to 'outputs/' directory")
    print()
    
    # Step 8: Export results
    print("8. Exporting results...")
    
    # Create outputs directory if it doesn't exist
    import os
    os.makedirs('outputs', exist_ok=True)
    
    # Export base forecast
    forecaster.export_forecast(
        base_forecast, 
        'outputs/base_forecast.xlsx', 
        format='excel'
    )
    
    # Export scenario forecasts
    with pd.ExcelWriter('outputs/scenario_forecasts.xlsx', engine='xlsxwriter') as writer:
        for scenario_name, scenario_df in scenario_forecasts.items():
            scenario_df.to_excel(writer, sheet_name=scenario_name, index=False)
    
    # Export vintage analysis
    vintage_metrics.to_excel('outputs/vintage_analysis.xlsx', index=False)
    
    print("   Results exported to 'outputs/' directory")
    print()
    
    # Step 9: Summary report
    print("=== FORECAST SUMMARY ===")
    print(f"Portfolio Size: {portfolio_df['loan_amount'].sum():,.0f}")
    print(f"Number of Vintages: {len(portfolio_df)}")
    print(f"Forecast Horizon: {forecast_end_date.year - current_date.year} years")
    print(f"Total Forecasted Charge-offs: ${base_forecast['charge_off_amount'].sum():,.0f}")
    print(f"Lifetime Loss Rate: {base_forecast['charge_off_amount'].sum() / portfolio_df['loan_amount'].sum():.2%}")
    
    # Risk metrics
    print("\n=== RISK METRICS ===")
    print(f"Peak Monthly Charge-off Rate: {base_forecast['charge_off_rate'].max():.2%}")
    print(f"Average Monthly Charge-off Rate: {base_forecast['charge_off_rate'].mean():.2%}")
    print(f"Charge-off Volatility: {base_forecast['charge_off_rate'].std():.2%}")
    
    # Scenario analysis
    print("\n=== SCENARIO ANALYSIS ===")
    for scenario_name, scenario_df in scenario_forecasts.items():
        lifetime_loss_rate = scenario_df['charge_off_amount'].sum() / portfolio_df['loan_amount'].sum()
        print(f"{scenario_name}: {lifetime_loss_rate:.2%} lifetime loss rate")
    
    print("\n=== FORECAST COMPLETE ===")
    print("Check the 'outputs/' directory for detailed results and visualizations.")


if __name__ == "__main__":
    main() 