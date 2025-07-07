#!/usr/bin/env python3
"""
Demonstration of flexible data handling in the Vintage Forecasts system.
Shows how the system processes both complete and incomplete vintage data formats.
For more details, see the methodology and main example scripts.
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_loader import DataLoader
from vintage_analyzer import VintageAnalyzer

def demonstrate_flexible_data_handling():
    """Demonstrate the flexible data handling feature."""
    
    print("="*80)
    print("FLEXIBLE DATA HANDLING DEMONSTRATION")
    print("="*80)
    
    # Initialize data loader
    data_loader = DataLoader()
    
    # Scenario 1: Incomplete Vintage Data (loans disappear after charge-off)
    print("\n1. INCOMPLETE VINTAGE DATA SCENARIO")
    print("-" * 50)
    
    print("Generating incomplete vintage data (loans disappear after charge-off)...")
    incomplete_data = data_loader.generate_synthetic_data(
        num_vintages=2,
        loans_per_vintage=30,
        max_seasoning=12,
        incomplete_vintages=True
    )
    
    print(f"   Generated {len(incomplete_data)} records")
    
    # Show characteristics of incomplete data
    loan_counts = incomplete_data.groupby('loan_id').size()
    print(f"   Average seasoning months per loan: {loan_counts.mean():.1f}")
    print(f"   Loans with incomplete seasoning: {sum(loan_counts < 12)}")
    
    # Sample loan analysis
    sample_loan = incomplete_data['loan_id'].iloc[0]
    sample_loan_data = incomplete_data[incomplete_data['loan_id'] == sample_loan]
    charge_off_month = sample_loan_data[sample_loan_data['charge_off_flag'] == 1]['seasoning_month']
    
    print(f"   Sample loan {sample_loan}:")
    print(f"     - Seasoning months: {len(sample_loan_data)}")
    print(f"     - Charge-off month: {charge_off_month.iloc[0] if not charge_off_month.empty else 'None'}")
    
    # Complete the data automatically
    print("\n   Completing vintage data automatically...")
    data_loader.data = incomplete_data
    completed_data = data_loader.preprocess_data()
    
    print(f"   Completed data: {len(completed_data)} records")
    
    # Show the same loan after completion
    completed_sample_data = completed_data[completed_data['loan_id'] == sample_loan]
    print(f"   Same loan after completion:")
    print(f"     - Seasoning months: {len(completed_sample_data)}")
    print(f"     - Charge-off months: {list(completed_sample_data[completed_sample_data['charge_off_flag'] == 1]['seasoning_month'])}")
    
    # Scenario 2: Complete Vintage Data (traditional approach)
    print("\n2. COMPLETE VINTAGE DATA SCENARIO")
    print("-" * 50)
    
    print("Generating complete vintage data (loans continue after charge-off)...")
    complete_data = data_loader.generate_synthetic_data(
        num_vintages=2,
        loans_per_vintage=30,
        max_seasoning=12,
        incomplete_vintages=False
    )
    
    print(f"   Generated {len(complete_data)} records")
    
    # Show characteristics of complete data
    complete_loan_counts = complete_data.groupby('loan_id').size()
    print(f"   Average seasoning months per loan: {complete_loan_counts.mean():.1f}")
    print(f"   All loans have complete seasoning: {all(complete_loan_counts == 12)}")
    
    # Process complete data
    print("\n   Processing complete data...")
    data_loader.data = complete_data
    processed_complete_data = data_loader.preprocess_data()
    
    print(f"   Processed data: {len(processed_complete_data)} records")
    
    # Comparison
    print("\n3. COMPARISON")
    print("-" * 50)
    
    print("Data Volume Comparison:")
    print(f"   Incomplete approach: {len(incomplete_data)} → {len(completed_data)} records")
    print(f"   Complete approach: {len(complete_data)} → {len(processed_complete_data)} records")
    print(f"   Volume reduction with incomplete: {((len(complete_data) - len(incomplete_data)) / len(complete_data) * 100):.1f}%")
    
    # Analysis capability
    print("\nAnalysis Capability:")
    analyzer = VintageAnalyzer()
    
    # Test analysis on both approaches
    try:
        analysis_incomplete = analyzer.analyze_vintage_data(completed_data)
        print("   ✓ Incomplete data approach: Analysis successful")
        print(f"     FICO bands analyzed: {list(analysis_incomplete['fico_band_analysis'].keys())}")
    except Exception as e:
        print(f"   ✗ Incomplete data approach: Analysis failed - {e}")
    
    try:
        analysis_complete = analyzer.analyze_vintage_data(processed_complete_data)
        print("   ✓ Complete data approach: Analysis successful")
        print(f"     FICO bands analyzed: {list(analysis_complete['fico_band_analysis'].keys())}")
    except Exception as e:
        print(f"   ✗ Complete data approach: Analysis failed - {e}")
    
    # Benefits summary
    print("\n4. BENEFITS OF FLEXIBLE DATA HANDLING")
    print("-" * 50)
    
    print("✓ Handles both data formats automatically")
    print("✓ Reduces data volume with incomplete approach")
    print("✓ Maintains analysis accuracy")
    print("✓ No changes needed to analysis workflow")
    print("✓ Automatic detection and completion of missing data")
    print("✓ Preserves charge-off timing and amounts")
    
    print("\n" + "="*80)
    print("DEMONSTRATION COMPLETED SUCCESSFULLY!")
    print("="*80)

if __name__ == "__main__":
    demonstrate_flexible_data_handling() 