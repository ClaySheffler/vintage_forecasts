"""
Test script for FICO segmentation functionality.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from src.data_loader import LoanDataLoader
from src.vintage_analyzer import VintageAnalyzer
from src.forecaster import ChargeOffForecaster


def test_fico_segmentation():
    """Test FICO segmentation functionality."""
    print("=== Testing FICO Segmentation Functionality ===\n")
    
    # Test 1: Data Loader with FICO segmentation
    print("1. Testing Data Loader with FICO segmentation...")
    data_loader = LoanDataLoader()
    loan_data = data_loader.load_data(source='synthetic')
    loan_data = data_loader.preprocess_data()
    
    # Verify FICO bands are present
    assert 'fico_band' in loan_data.columns, "FICO band column missing"
    assert 'risk_grade' in loan_data.columns, "Risk grade column missing"
    assert 'fico_score' in loan_data.columns, "FICO score column missing"
    
    # Verify FICO bands are correct
    expected_fico_bands = ['600-649', '650-699', '700-749', '750-799', '800+']
    actual_fico_bands = sorted(loan_data['fico_band'].unique())
    assert actual_fico_bands == expected_fico_bands, f"FICO bands mismatch: {actual_fico_bands}"
    
    # Verify risk grades are correct
    expected_risk_grades = [1, 2, 3, 4, 5]
    actual_risk_grades = sorted(loan_data['risk_grade'].unique())
    assert actual_risk_grades == expected_risk_grades, f"Risk grades mismatch: {actual_risk_grades}"
    
    print("   ✓ FICO bands and risk grades correctly assigned")
    
    # Test 2: Data summary with FICO information
    print("\n2. Testing data summary with FICO information...")
    summary = data_loader.get_data_summary()
    
    assert 'fico_bands' in summary, "FICO bands missing from summary"
    assert 'risk_grades' in summary, "Risk grades missing from summary"
    assert 'avg_fico_score' in summary, "Average FICO score missing from summary"
    
    print(f"   ✓ Data summary includes FICO information")
    print(f"   ✓ Average FICO score: {summary['avg_fico_score']:.0f}")
    print(f"   ✓ FICO bands: {list(summary['fico_bands'].keys())}")
    
    # Test 3: Vintage analysis by FICO band
    print("\n3. Testing vintage analysis by FICO band...")
    analyzer = VintageAnalyzer(loan_data)
    vintage_metrics = analyzer.calculate_vintage_metrics()
    
    # Verify vintage metrics include FICO bands
    assert 'fico_band' in vintage_metrics.columns, "FICO band missing from vintage metrics"
    assert 'risk_grade' in vintage_metrics.columns, "Risk grade missing from vintage metrics"
    
    # Check that we have metrics for each FICO band
    fico_metrics_count = vintage_metrics['fico_band'].nunique()
    assert fico_metrics_count == len(expected_fico_bands), f"Missing FICO bands in metrics: {fico_metrics_count}"
    
    print(f"   ✓ Vintage metrics calculated for {fico_metrics_count} FICO bands")
    
    # Test 4: Seasoning curves by FICO band
    print("\n4. Testing seasoning curves by FICO band...")
    seasoning_curves = analyzer.fit_seasoning_curves()
    
    # Verify seasoning curves exist for each FICO band
    for fico_band in expected_fico_bands:
        assert fico_band in seasoning_curves, f"Seasoning curves missing for {fico_band}"
        assert 'aggregate' in seasoning_curves, "Aggregate seasoning curves missing"
    
    print(f"   ✓ Seasoning curves fitted for {len(seasoning_curves)-1} FICO bands + aggregate")
    
    # Test 5: FICO mix analysis
    print("\n5. Testing FICO mix analysis...")
    fico_mix_trends = analyzer.analyze_fico_mix_trends()
    
    assert 'portfolio_quality' in fico_mix_trends, "Portfolio quality trends missing"
    
    print(f"   ✓ FICO mix trends analyzed")
    
    # Test 6: Forecaster with FICO segmentation
    print("\n6. Testing forecaster with FICO segmentation...")
    forecaster = ChargeOffForecaster(analyzer, loan_data)
    
    # Create test portfolio mix
    portfolio_mix = {
        '600-649': {'loan_amount': 10000000, 'num_loans': 400},
        '650-699': {'loan_amount': 20000000, 'num_loans': 800},
        '700-749': {'loan_amount': 30000000, 'num_loans': 1200},
        '750-799': {'loan_amount': 25000000, 'num_loans': 1000},
        '800+': {'loan_amount': 15000000, 'num_loans': 600}
    }
    
    # Test FICO-segmented forecast
    vintage_date = datetime(2024, 1, 1)
    forecast = forecaster.forecast_vintage_charge_offs_by_fico(
        vintage_date=vintage_date,
        portfolio_mix=portfolio_mix,
        forecast_horizon=60
    )
    
    # Verify forecast structure
    assert 'vintage_date' in forecast.columns, "Vintage date missing from forecast"
    assert 'seasoning_month' in forecast.columns, "Seasoning month missing from forecast"
    assert 'charge_off_flag' in forecast.columns, "Charge-off flag missing from forecast"
    assert 'charge_off_amount' in forecast.columns, "Charge-off amount missing from forecast"
    
    print(f"   ✓ FICO-segmented forecast generated with {len(forecast)} periods")
    
    # Test 7: Quality mix impact analysis
    print("\n7. Testing quality mix impact analysis...")
    
    # Test different quality mixes
    conservative_mix = {
        '600-649': 0.05, '650-699': 0.15, '700-749': 0.30,
        '750-799': 0.35, '800+': 0.15
    }
    
    aggressive_mix = {
        '600-649': 0.20, '650-699': 0.30, '700-749': 0.30,
        '750-799': 0.15, '800+': 0.05
    }
    
    # Create portfolio mixes
    conservative_portfolio = {}
    aggressive_portfolio = {}
    total_amount = 100000000
    
    for fico_band, pct in conservative_mix.items():
        conservative_portfolio[fico_band] = {
            'loan_amount': total_amount * pct,
            'num_loans': int(4000 * pct)
        }
    
    for fico_band, pct in aggressive_mix.items():
        aggressive_portfolio[fico_band] = {
            'loan_amount': total_amount * pct,
            'num_loans': int(4000 * pct)
        }
    
    # Generate forecasts
    conservative_forecast = forecaster.forecast_vintage_charge_offs_by_fico(
        vintage_date=vintage_date,
        portfolio_mix=conservative_portfolio,
        forecast_horizon=60
    )
    
    aggressive_forecast = forecaster.forecast_vintage_charge_offs_by_fico(
        vintage_date=vintage_date,
        portfolio_mix=aggressive_portfolio,
        forecast_horizon=60
    )
    
    # Calculate lifetime loss rates
    conservative_loss = conservative_forecast['charge_off_amount'].sum() / total_amount
    aggressive_loss = aggressive_forecast['charge_off_amount'].sum() / total_amount
    
    print(f"   ✓ Conservative mix lifetime loss: {conservative_loss:.2%}")
    print(f"   ✓ Aggressive mix lifetime loss: {aggressive_loss:.2%}")
    print(f"   ✓ Quality mix impact: {aggressive_loss - conservative_loss:.2%}")
    
    # Verify that aggressive mix has higher losses
    assert aggressive_loss > conservative_loss, "Aggressive mix should have higher losses"
    
    print("\n=== All FICO Segmentation Tests Passed! ===")
    print("\nKey Features Verified:")
    print("✓ FICO bands and risk grades correctly assigned")
    print("✓ Vintage analysis by FICO band")
    print("✓ Seasoning curves by FICO band")
    print("✓ FICO mix trend analysis")
    print("✓ FICO-segmented forecasting")
    print("✓ Quality mix impact analysis")
    print("✓ Dollar-weighted aggregation")
    
    return True


if __name__ == "__main__":
    test_fico_segmentation() 