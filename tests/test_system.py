"""
Simple test script to verify the vintage forecasting system works.
"""

def test_imports():
    """Test that all modules can be imported successfully."""
    try:
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
        from scipy import stats
        from scipy.optimize import curve_fit
        print("✓ All required libraries imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_data_loader():
    """Test the data loader functionality."""
    try:
        from src.data_loader import LoanDataLoader
        
        # Test data loader
        data_loader = LoanDataLoader()
        loan_data = data_loader.load_sample_data()
        
        # Check data structure
        required_columns = ['loan_id', 'vintage_date', 'report_date', 'seasoning_month', 
                           'loan_amount', 'charge_off_rate', 'charge_off_amount']
        
        missing_columns = [col for col in required_columns if col not in loan_data.columns]
        if missing_columns:
            print(f"✗ Missing columns: {missing_columns}")
            return False
        
        print(f"✓ Data loader test passed - {len(loan_data):,} records loaded")
        return True
        
    except Exception as e:
        print(f"✗ Data loader test failed: {e}")
        return False

def test_vintage_analyzer():
    """Test the vintage analyzer functionality."""
    try:
        from src.data_loader import LoanDataLoader
        from src.vintage_analyzer import VintageAnalyzer
        
        # Load sample data
        data_loader = LoanDataLoader()
        loan_data = data_loader.load_sample_data()
        loan_data = data_loader.preprocess_data()
        
        # Test vintage analyzer
        vintage_analyzer = VintageAnalyzer(loan_data)
        vintage_metrics = vintage_analyzer.calculate_vintage_metrics()
        seasoning_curves = vintage_analyzer.fit_seasoning_curves()
        
        # Check results
        if len(vintage_metrics) == 0:
            print("✗ No vintage metrics calculated")
            return False
        
        if not seasoning_curves:
            print("✗ No seasoning curves fitted")
            return False
        
        print(f"✓ Vintage analyzer test passed - {len(vintage_metrics)} vintage metrics calculated")
        return True
        
    except Exception as e:
        print(f"✗ Vintage analyzer test failed: {e}")
        return False

def test_forecaster():
    """Test the forecaster functionality."""
    try:
        from src.data_loader import LoanDataLoader
        from src.vintage_analyzer import VintageAnalyzer
        from src.forecaster import ChargeOffForecaster
        from datetime import datetime
        
        # Load and analyze data
        data_loader = LoanDataLoader()
        loan_data = data_loader.load_sample_data()
        loan_data = data_loader.preprocess_data()
        
        vintage_analyzer = VintageAnalyzer(loan_data)
        vintage_analyzer.calculate_vintage_metrics()
        vintage_analyzer.fit_seasoning_curves()
        
        # Test forecaster
        forecaster = ChargeOffForecaster(vintage_analyzer, loan_data)
        
        # Test single vintage forecast
        test_forecast = forecaster.forecast_vintage_charge_offs(
            vintage_date=datetime(2024, 1, 1),
            loan_amount=100_000_000,
            num_loans=3000,
            forecast_horizon=60
        )
        
        if len(test_forecast) == 0:
            print("✗ No forecast generated")
            return False
        
        print(f"✓ Forecaster test passed - {len(test_forecast)} forecast periods generated")
        return True
        
    except Exception as e:
        print(f"✗ Forecaster test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("=== Vintage Forecasting System Test ===\n")
    
    tests = [
        ("Library Imports", test_imports),
        ("Data Loader", test_data_loader),
        ("Vintage Analyzer", test_vintage_analyzer),
        ("Forecaster", test_forecaster)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"Running {test_name} test...")
        if test_func():
            passed += 1
        print()
    
    print(f"=== Test Results ===")
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! The vintage forecasting system is working correctly.")
        print("\nTo run the full system:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Run main script: python main.py")
        print("3. Or run interactive demo: jupyter notebook notebooks/vintage_forecasting_demo.ipynb")
    else:
        print("❌ Some tests failed. Please check the error messages above.")
    
    return passed == total

if __name__ == "__main__":
    main() 