"""
Main script for vintage charge-off forecasting with FICO segmentation.
Demonstrates the complete workflow from data loading to forecasting with quality mix analysis.
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
    Main function demonstrating the vintage charge-off forecasting workflow with FICO segmentation.
    """
    print("=== Vintage Charge-off Forecasting System with FICO Segmentation ===\n")
    
    # Step 1: Load and preprocess data
    print("1. Loading and preprocessing loan performance data...")
    data_loader = LoanDataLoader()
    
    # Choose data source: 'synthetic' or 'file'
    # For demonstration, we'll use synthetic data
    loan_data = data_loader.load_data(source='synthetic')
    loan_data = data_loader.preprocess_data()
    
    # Display data summary
    data_summary = data_loader.get_data_summary()
    print(f"   Loaded {data_summary['total_records']:,} loan performance records")
    print(f"   Date range: {data_summary['date_range']['start']} to {data_summary['date_range']['end']}")
    print(f"   Total loans: {data_summary['total_loans']:,}")
    print(f"   Total loan amount: ${data_summary['total_loan_amount']:,.0f}")
    print(f"   Average FICO score: {data_summary['avg_fico_score']:.0f}")
    print(f"   Average charge-off rate: {data_summary['avg_charge_off_rate']:.2%}")
    
    print("\n   FICO Band Distribution:")
    for fico_band, count in data_summary['fico_bands'].items():
        pct = count / data_summary['total_records'] * 100
        print(f"     {fico_band}: {count:,} records ({pct:.1f}%)")
    
    print("\n   Risk Grade Distribution:")
    for risk_grade, count in data_summary['risk_grades'].items():
        pct = count / data_summary['total_records'] * 100
        print(f"     Grade {risk_grade}: {count:,} records ({pct:.1f}%)")
    
    # Step 2: Perform vintage analysis with FICO segmentation
    print("\n2. Performing vintage analysis with FICO segmentation...")
    vintage_analyzer = VintageAnalyzer(loan_data)
    vintage_metrics = vintage_analyzer.calculate_vintage_metrics()
    seasoning_curves = vintage_analyzer.fit_seasoning_curves()
    
    print("   Fitted seasoning curves by FICO band:")
    for fico_band, curves in seasoning_curves.items():
        if fico_band != 'aggregate':
            print(f"     {fico_band}:")
            for curve_name, curve_info in curves.items():
                if curve_info is not None:
                    print(f"       {curve_name}: R² = {curve_info['r_squared']:.3f}")
    
    # Analyze FICO mix trends
    fico_mix_trends = vintage_analyzer.analyze_fico_mix_trends()
    print(f"\n   Analyzed FICO mix trends for {len(fico_mix_trends)} vintage periods")
    
    # Analyze vintage patterns by FICO band
    vintage_patterns = vintage_analyzer.identify_vintage_patterns()
    print(f"   Identified vintage patterns for {len(vintage_analyzer.fico_bands)} FICO bands")
    
    # Step 3: Create forecaster
    print("3. Initializing charge-off forecaster...")
    forecaster = ChargeOffForecaster(vintage_analyzer, loan_data)
    
    # Step 4: Generate portfolio forecast with FICO segmentation
    print("\n4. Generating portfolio charge-off forecast with FICO segmentation...")
    
    # Create sample portfolio data with FICO segmentation for forecasting
    portfolio_data = []
    current_date = datetime(2024, 1, 1)
    
    # Define FICO band mix scenarios
    fico_mix_scenarios = {
        'Conservative': {  # Higher quality mix
            '600-649': 0.05,   # 5% Very High Risk
            '650-699': 0.15,   # 15% High Risk
            '700-749': 0.30,   # 30% Medium Risk
            '750-799': 0.35,   # 35% Low Risk
            '800+': 0.15       # 15% Very Low Risk
        },
        'Balanced': {  # Balanced mix
            '600-649': 0.10,   # 10% Very High Risk
            '650-699': 0.20,   # 20% High Risk
            '700-749': 0.35,   # 35% Medium Risk
            '750-799': 0.25,   # 25% Low Risk
            '800+': 0.10       # 10% Very Low Risk
        },
        'Aggressive': {  # Lower quality mix
            '600-649': 0.20,   # 20% Very High Risk
            '650-699': 0.30,   # 30% High Risk
            '700-749': 0.30,   # 30% Medium Risk
            '750-799': 0.15,   # 15% Low Risk
            '800+': 0.05       # 5% Very Low Risk
        }
    }
    
    # Generate monthly vintages for the next 12 months with different quality mixes
    for i in range(12):
        vintage_date = current_date + timedelta(days=30*i)
        total_loan_amount = np.random.uniform(50_000_000, 100_000_000)  # $50M-$100M per month
        total_num_loans = np.random.randint(2000, 5000)
        
        # Choose mix scenario based on vintage (simulate quality shifts over time)
        if i < 4:
            mix_scenario = 'Conservative'  # Early vintages - higher quality
        elif i < 8:
            mix_scenario = 'Balanced'      # Middle vintages - balanced
        else:
            mix_scenario = 'Aggressive'    # Later vintages - lower quality
        
        fico_mix = fico_mix_scenarios[mix_scenario]
        
        # Generate loans for each FICO band
        for fico_band, mix_pct in fico_mix.items():
            loan_amount = total_loan_amount * mix_pct
            num_loans = int(total_num_loans * mix_pct)
            
            if num_loans > 0:  # Only add if there are loans in this band
                portfolio_data.append({
                    'vintage_date': vintage_date,
                    'fico_band': fico_band,
                    'loan_amount': loan_amount,
                    'num_loans': num_loans,
                    'mix_scenario': mix_scenario
                })
    
    portfolio_df = pd.DataFrame(portfolio_data)
    forecast_end_date = datetime(2034, 12, 31)  # 10-year forecast horizon
    
    print(f"   Created portfolio with {len(portfolio_df)} vintage-FICO combinations")
    print(f"   Total portfolio size: ${portfolio_df['loan_amount'].sum():,.0f}")
    print(f"   Forecast horizon: {forecast_end_date.year - current_date.year} years")
    
    # Analyze portfolio mix
    print("\n   Portfolio Quality Mix Analysis:")
    mix_analysis = portfolio_df.groupby('mix_scenario').agg({
        'loan_amount': 'sum',
        'num_loans': 'sum'
    }).reset_index()
    
    for _, row in mix_analysis.iterrows():
        scenario = row['mix_scenario']
        amount = row['loan_amount']
        loans = row['num_loans']
        print(f"     {scenario}: ${amount:,.0f} ({loans:,} loans)")
    
    # Generate base forecast
    base_forecast = forecaster.forecast_portfolio_charge_offs(
        portfolio_data=portfolio_df,
        forecast_end_date=forecast_end_date,
        vintage_quality_model=vintage_patterns
    )
    
    print(f"\n   Forecast Results:")
    print(f"     Forecast period: {base_forecast['report_date'].min()} to {base_forecast['report_date'].max()}")
    print(f"     Total forecasted charge-offs: ${base_forecast['charge_off_amount'].sum():,.0f}")
    print(f"     Peak charge-off rate: {base_forecast['charge_off_rate'].max():.2%}")
    print(f"     Average charge-off rate: {base_forecast['charge_off_rate'].mean():.2%}")
    
    # Step 5: Generate scenario forecasts
    print("\n5. Generating scenario forecasts...")
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
        lifetime_loss_rate = total_charge_offs / portfolio_df['loan_amount'].sum()
        print(f"     {scenario_name}: ${total_charge_offs:,.0f} total, {peak_rate:.2%} peak rate, {lifetime_loss_rate:.2%} lifetime loss")
    
    # Step 6: Calculate and display key metrics
    print("\n6. Calculating forecast metrics...")
    metrics = forecaster.calculate_forecast_metrics(base_forecast)
    
    print("   Key Forecast Metrics:")
    print(f"     Total Charge-offs: ${metrics['total_charge_offs']:,.0f}")
    print(f"     Peak Charge-off Rate: {metrics['peak_charge_off_rate']:.2%}")
    print(f"     Peak Month: {metrics['peak_charge_off_month'].strftime('%Y-%m')}")
    print(f"     Average Charge-off Rate: {metrics['avg_charge_off_rate']:.2%}")
    
    if 'median_charge_off_timing' in metrics:
        print(f"     Median Charge-off Timing: {metrics['median_charge_off_timing'].strftime('%Y-%m')}")
    
    # Step 7: Analyze quality mix impact
    print("\n7. Analyzing quality mix impact on forecasts...")
    
    # Compare different quality mix scenarios
    mix_impact_analysis = {}
    for scenario_name, fico_mix in fico_mix_scenarios.items():
        # Create portfolio with this mix
        scenario_portfolio = []
        for i in range(6):  # 6 months for comparison
            vintage_date = current_date + timedelta(days=30*i)
            total_loan_amount = 75_000_000  # Fixed amount for comparison
            total_num_loans = 3000
            
            for fico_band, mix_pct in fico_mix.items():
                loan_amount = total_loan_amount * mix_pct
                num_loans = int(total_num_loans * mix_pct)
                
                if num_loans > 0:
                    scenario_portfolio.append({
                        'vintage_date': vintage_date,
                        'fico_band': fico_band,
                        'loan_amount': loan_amount,
                        'num_loans': num_loans
                    })
        
        scenario_df = pd.DataFrame(scenario_portfolio)
        
        # Generate forecast for this scenario
        scenario_forecast = forecaster.forecast_portfolio_charge_offs(
            portfolio_data=scenario_df,
            forecast_end_date=datetime(2030, 12, 31),
            vintage_quality_model=vintage_patterns
        )
        
        total_charge_offs = scenario_forecast['charge_off_amount'].sum()
        lifetime_loss_rate = total_charge_offs / scenario_df['loan_amount'].sum()
        peak_rate = scenario_forecast['charge_off_rate'].max()
        
        mix_impact_analysis[scenario_name] = {
            'total_charge_offs': total_charge_offs,
            'lifetime_loss_rate': lifetime_loss_rate,
            'peak_rate': peak_rate
        }
    
    print("   Quality Mix Impact Analysis:")
    for scenario_name, metrics in mix_impact_analysis.items():
        print(f"     {scenario_name}: ${metrics['total_charge_offs']:,.0f} total, "
              f"{metrics['peak_rate']:.2%} peak rate, {metrics['lifetime_loss_rate']:.2%} lifetime loss")
    
    # Step 8: Create visualizations
    print("\n8. Creating visualizations...")
    
    # Create outputs directory if it doesn't exist
    import os
    os.makedirs('outputs', exist_ok=True)
    
    # Vintage analysis plots
    vintage_analyzer.plot_vintage_analysis(save_path='outputs/vintage_analysis_fico.png')
    
    # Forecast plots
    forecaster.plot_forecast_results(
        base_forecast, 
        scenario_forecasts, 
        save_path='outputs/forecast_results_fico.png'
    )
    
    print("   Plots saved to 'outputs/' directory")
    
    # Step 9: Export results
    print("\n9. Exporting results...")
    
    # Export base forecast
    forecaster.export_forecast(
        base_forecast, 
        'outputs/base_forecast_fico.xlsx', 
        format='excel'
    )
    
    # Export scenario forecasts
    with pd.ExcelWriter('outputs/scenario_forecasts_fico.xlsx', engine='xlsxwriter') as writer:
        for scenario_name, scenario_df in scenario_forecasts.items():
            scenario_df.to_excel(writer, sheet_name=scenario_name, index=False)
    
    # Export vintage analysis
    vintage_metrics.to_excel('outputs/vintage_analysis_fico.xlsx', index=False)
    
    # Export FICO mix analysis
    fico_mix_by_vintage = data_loader.get_fico_mix_by_vintage()
    fico_mix_by_vintage.to_excel('outputs/fico_mix_analysis.xlsx', index=False)
    
    print("   Results exported to 'outputs/' directory")
    
    # Step 10: Summary report
    print("\n=== FORECAST SUMMARY WITH FICO SEGMENTATION ===")
    print(f"Portfolio Size: {portfolio_df['loan_amount'].sum():,.0f}")
    print(f"Number of Vintage-FICO Combinations: {len(portfolio_df)}")
    print(f"FICO Bands: {', '.join(forecaster.fico_bands)}")
    print(f"Forecast Horizon: {forecast_end_date.year - current_date.year} years")
    print(f"Total Forecasted Charge-offs: ${base_forecast['charge_off_amount'].sum():,.0f}")
    print(f"Lifetime Loss Rate: {base_forecast['charge_off_amount'].sum() / portfolio_df['loan_amount'].sum():.2%}")
    
    # Risk metrics
    print("\n=== RISK METRICS ===")
    print(f"Peak Monthly Charge-off Rate: {base_forecast['charge_off_rate'].max():.2%}")
    print(f"Average Monthly Charge-off Rate: {base_forecast['charge_off_rate'].mean():.2%}")
    print(f"Charge-off Volatility: {base_forecast['charge_off_rate'].std():.2%}")
    
    # Quality mix impact
    print("\n=== QUALITY MIX IMPACT ===")
    conservative_loss = mix_impact_analysis['Conservative']['lifetime_loss_rate']
    aggressive_loss = mix_impact_analysis['Aggressive']['lifetime_loss_rate']
    impact = aggressive_loss - conservative_loss
    print(f"Conservative Mix Lifetime Loss: {conservative_loss:.2%}")
    print(f"Aggressive Mix Lifetime Loss: {aggressive_loss:.2%}")
    print(f"Quality Mix Impact: {impact:.2%} difference")
    
    # Scenario analysis
    print("\n=== SCENARIO ANALYSIS ===")
    for scenario_name, scenario_df in scenario_forecasts.items():
        lifetime_loss_rate = scenario_df['charge_off_amount'].sum() / portfolio_df['loan_amount'].sum()
        print(f"{scenario_name}: {lifetime_loss_rate:.2%} lifetime loss rate")
    
    print("\n=== KEY INSIGHTS ===")
    print("1. FICO segmentation provides granular risk analysis by credit quality")
    print("2. Quality mix shifts significantly impact charge-off projections")
    print("3. Each FICO band has distinct seasoning patterns and vintage performance")
    print("4. Dollar-weighted aggregation ensures accurate portfolio-level forecasts")
    print("5. Scenario analysis captures both economic and quality mix risks")
    
    print("\n=== FORECASTING COMPLETE ===")
    print("The FICO-segmented vintage forecasting system provides comprehensive")
    print("insights into portfolio risk with quality mix analysis.")


if __name__ == "__main__":
    main() 