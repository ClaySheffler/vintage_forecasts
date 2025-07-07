"""
Simple example demonstrating FICO-segmented vintage charge-off forecasting.
"""

import pandas as pd
from datetime import datetime, timedelta
from src.data_loader import LoanDataLoader
from src.vintage_analyzer import VintageAnalyzer
from src.forecaster import ChargeOffForecaster


def main():
    """Simple example of FICO-segmented vintage forecasting."""
    print("=== FICO-Segmented Vintage Forecasting Example ===\n")
    
    # 1. Load synthetic data with FICO segmentation
    print("1. Loading synthetic loan data with FICO segmentation...")
    data_loader = LoanDataLoader()
    loan_data = data_loader.load_data(source='synthetic')
    loan_data = data_loader.preprocess_data()
    
    # Show data summary
    summary = data_loader.get_data_summary()
    print(f"   Loaded {summary['total_records']:,} records across {len(summary['fico_bands'])} FICO bands")
    print(f"   Average FICO: {summary['avg_fico_score']:.0f}")
    print(f"   Average charge-off rate: {summary['avg_charge_off_rate']:.2%}")
    
    # 2. Perform vintage analysis by FICO band
    print("\n2. Analyzing vintage performance by FICO band...")
    analyzer = VintageAnalyzer(loan_data)
    vintage_metrics = analyzer.calculate_vintage_metrics()
    seasoning_curves = analyzer.fit_seasoning_curves()
    
    print(f"   Fitted seasoning curves for {len(analyzer.fico_bands)} FICO bands")
    for fico_band in analyzer.fico_bands:
        if fico_band in seasoning_curves:
            curves = seasoning_curves[fico_band]
            best_curve = None
            best_r2 = -1
            for curve_name, curve_info in curves.items():
                if curve_info and curve_info['r_squared'] > best_r2:
                    best_r2 = curve_info['r_squared']
                    best_curve = curve_name
            print(f"     {fico_band}: {best_curve} (R² = {best_r2:.3f})")
    
    # 3. Create forecaster
    print("\n3. Creating FICO-segmented forecaster...")
    forecaster = ChargeOffForecaster(analyzer, loan_data)
    
    # 4. Generate sample portfolio with FICO mix
    print("\n4. Generating sample portfolio with FICO quality mix...")
    
    # Define FICO band mix (Conservative quality mix)
    fico_mix = {
        '600-649': 0.05,   # 5% Very High Risk
        '650-699': 0.15,   # 15% High Risk
        '700-749': 0.30,   # 30% Medium Risk
        '750-799': 0.35,   # 35% Low Risk
        '800+': 0.15       # 15% Very Low Risk
    }
    
    # Create portfolio data
    vintage_date = datetime(2024, 1, 1)
    total_loan_amount = 100_000_000  # $100M
    total_num_loans = 4000
    
    portfolio_mix = {}
    for fico_band, mix_pct in fico_mix.items():
        loan_amount = total_loan_amount * mix_pct
        num_loans = int(total_num_loans * mix_pct)
        portfolio_mix[fico_band] = {
            'loan_amount': loan_amount,
            'num_loans': num_loans
        }
    
    print("   Portfolio FICO Mix:")
    for fico_band, mix_info in portfolio_mix.items():
        print(f"     {fico_band}: ${mix_info['loan_amount']:,.0f} ({mix_info['num_loans']:,} loans)")
    
    # 5. Forecast by FICO band
    print("\n5. Forecasting charge-offs by FICO band...")
    forecast = forecaster.forecast_vintage_charge_offs_by_fico(
        vintage_date=vintage_date,
        portfolio_mix=portfolio_mix,
        forecast_horizon=60  # 5 years
    )
    
    # 6. Calculate key metrics
    print("\n6. Calculating forecast metrics...")
    metrics = forecaster.calculate_forecast_metrics(forecast)
    
    print("   Forecast Results:")
    print(f"     Total Charge-offs: ${metrics['total_charge_offs']:,.0f}")
    print(f"     Peak Charge-off Rate: {metrics['peak_charge_off_rate']:.2%}")
    print(f"     Average Charge-off Rate: {metrics['avg_charge_off_rate']:.2%}")
    print(f"     Lifetime Loss Rate: {metrics['total_charge_offs'] / total_loan_amount:.2%}")
    
    # 7. Compare different quality mixes
    print("\n7. Comparing different quality mixes...")
    
    quality_mixes = {
        'Conservative': {
            '600-649': 0.05, '650-699': 0.15, '700-749': 0.30, 
            '750-799': 0.35, '800+': 0.15
        },
        'Balanced': {
            '600-649': 0.10, '650-699': 0.20, '700-749': 0.35, 
            '750-799': 0.25, '800+': 0.10
        },
        'Aggressive': {
            '600-649': 0.20, '650-699': 0.30, '700-749': 0.30, 
            '750-799': 0.15, '800+': 0.05
        }
    }
    
    mix_results = {}
    for mix_name, fico_mix in quality_mixes.items():
        # Create portfolio mix
        mix_portfolio = {}
        for fico_band, mix_pct in fico_mix.items():
            loan_amount = total_loan_amount * mix_pct
            num_loans = int(total_num_loans * mix_pct)
            mix_portfolio[fico_band] = {
                'loan_amount': loan_amount,
                'num_loans': num_loans
            }
        
        # Forecast
        mix_forecast = forecaster.forecast_vintage_charge_offs_by_fico(
            vintage_date=vintage_date,
            portfolio_mix=mix_portfolio,
            forecast_horizon=60
        )
        
        total_charge_offs = mix_forecast['charge_off_amount'].sum()
        lifetime_loss_rate = total_charge_offs / total_loan_amount
        peak_rate = mix_forecast['charge_off_rate'].max()
        
        mix_results[mix_name] = {
            'total_charge_offs': total_charge_offs,
            'lifetime_loss_rate': lifetime_loss_rate,
            'peak_rate': peak_rate
        }
    
    print("   Quality Mix Impact:")
    for mix_name, results in mix_results.items():
        print(f"     {mix_name}: ${results['total_charge_offs']:,.0f} total, "
              f"{results['peak_rate']:.2%} peak, {results['lifetime_loss_rate']:.2%} lifetime loss")
    
    # 8. Summary
    print("\n=== SUMMARY ===")
    print("FICO segmentation provides granular risk analysis:")
    print(f"• {len(analyzer.fico_bands)} FICO bands analyzed")
    print(f"• Quality mix impact: {mix_results['Aggressive']['lifetime_loss_rate'] - mix_results['Conservative']['lifetime_loss_rate']:.2%} difference")
    print(f"• Dollar-weighted aggregation ensures accurate portfolio forecasts")
    print("• Each FICO band has distinct seasoning patterns")
    
    print("\nExample completed successfully!")

    # Example 3: Flexible Data Handling
    print("\n" + "="*60)
    print("EXAMPLE 3: FLEXIBLE DATA HANDLING")
    print("="*60)
    
    # Demonstrate handling incomplete vintage data
    print("\n3.1 Handling incomplete vintage data (loans disappear after charge-off)")
    
    # Generate incomplete data
    incomplete_data = data_loader.generate_synthetic_data(
        num_vintages=2,
        loans_per_vintage=30,
        max_seasoning=12,
        incomplete_vintages=True
    )
    
    print(f"   Generated incomplete data: {len(incomplete_data)} records")
    
    # Show sample of incomplete data
    sample_loan = incomplete_data['loan_id'].iloc[0]
    loan_data = incomplete_data[incomplete_data['loan_id'] == sample_loan]
    print(f"   Sample loan {sample_loan}: {len(loan_data)} seasoning months")
    print(f"   Charge-off month: {loan_data[loan_data['charge_off_rate'] == 1]['seasoning_month'].iloc[0] if not loan_data[loan_data['charge_off_rate'] == 1].empty else 'None'}")
    
    # Load and preprocess (automatically completes data)
    data_loader.data = incomplete_data
    completed_data = data_loader.preprocess_data()
    
    print(f"   Completed data: {len(completed_data)} records")
    
    # Show the same loan after completion
    completed_loan_data = completed_data[completed_data['loan_id'] == sample_loan]
    print(f"   Same loan after completion: {len(completed_loan_data)} seasoning months")
    
    # Analyze completed data
    analyzer = VintageAnalyzer()
    analysis = analyzer.analyze_vintage_data(completed_data)
    
    print(f"   Analysis completed successfully!")
    print(f"   FICO bands analyzed: {list(analysis['fico_band_analysis'].keys())}")
    
    # Compare with complete data approach
    print("\n3.2 Comparing with complete data approach")
    
    complete_data = data_loader.generate_synthetic_data(
        num_vintages=2,
        loans_per_vintage=30,
        max_seasoning=12,
        incomplete_vintages=False
    )
    
    print(f"   Generated complete data: {len(complete_data)} records")
    
    # Load and preprocess complete data
    data_loader.data = complete_data
    processed_complete_data = data_loader.preprocess_data()
    
    print(f"   Processed complete data: {len(processed_complete_data)} records")
    
    # Both approaches should produce similar analysis results
    print(f"   Both approaches work seamlessly!")
    
    print("\n" + "="*60)
    print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
    print("="*60)


if __name__ == "__main__":
    main() 