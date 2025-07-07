    def test_flexible_data_handling(self):
        """Test flexible data handling for both complete and incomplete vintage data."""
        print("\nTesting flexible data handling...")
        
        # Test incomplete vintage data generation
        incomplete_data = self.data_loader.generate_synthetic_data(
            num_vintages=2,
            loans_per_vintage=20,
            max_seasoning=12,
            incomplete_vintages=True
        )
        
        # Verify incomplete data characteristics
        self.assertGreater(len(incomplete_data), 0)
        
        # Check that some loans have fewer seasoning months than max_seasoning
        loan_seasoning_counts = incomplete_data.groupby('loan_id')['seasoning_month'].max()
        self.assertTrue(any(loan_seasoning_counts < 12), "Some loans should have incomplete seasoning")
        
        # Test data completion
        self.data_loader.data = incomplete_data
        completed_data = self.data_loader.preprocess_data()
        
        # Verify data was completed
        self.assertGreaterEqual(len(completed_data), len(incomplete_data))
        
        # Check that all loans now have complete seasoning curves
        completed_loan_seasoning_counts = completed_data.groupby('loan_id')['seasoning_month'].max()
        self.assertTrue(all(completed_loan_seasoning_counts >= 12), "All loans should have complete seasoning")
        
        # Test complete vintage data generation
        complete_data = self.data_loader.generate_synthetic_data(
            num_vintages=2,
            loans_per_vintage=20,
            max_seasoning=12,
            incomplete_vintages=False
        )
        
        # Verify complete data characteristics
        self.assertGreater(len(complete_data), len(incomplete_data))
        
        # Check that all loans have full seasoning curves
        complete_loan_seasoning_counts = complete_data.groupby('loan_id')['seasoning_month'].max()
        self.assertTrue(all(complete_loan_seasoning_counts == 12), "All loans should have full seasoning")
        
        # Test that both approaches produce analyzable data
        self.data_loader.data = complete_data
        processed_complete_data = self.data_loader.preprocess_data()
        
        # Both should be analyzable
        analyzer = VintageAnalyzer()
        
        # Test analysis on completed data
        analysis_completed = analyzer.analyze_vintage_data(completed_data)
        self.assertIn('fico_band_analysis', analysis_completed)
        
        # Test analysis on complete data
        analysis_complete = analyzer.analyze_vintage_data(processed_complete_data)
        self.assertIn('fico_band_analysis', analysis_complete)
        
        print("   ✓ Flexible data handling works correctly")
    
    def test_charge_off_pattern_analysis(self):
        """Test charge-off pattern analysis functionality."""
        print("\nTesting charge-off pattern analysis...")
        
        # Generate test data
        test_data = self.data_loader.generate_synthetic_data(
            num_vintages=3,
            loans_per_vintage=50,
            max_seasoning=24,
            incomplete_vintages=True
        )
        
        # Complete the data
        self.data_loader.data = test_data
        completed_data = self.data_loader.preprocess_data()
        
        # Analyze charge-off patterns
        analyzer = VintageAnalyzer()
        analysis = analyzer.analyze_vintage_data(completed_data)
        
        # Verify charge-off pattern analysis exists
        self.assertIn('charge_off_patterns', analysis)
        
        # Check that patterns exist for each FICO band
        for fico_band in ['600-649', '650-699', '700-749']:
            if fico_band in analysis['charge_off_patterns']:
                patterns = analysis['charge_off_patterns'][fico_band]
                self.assertIn('cumulative_charge_off_curve', patterns)
                self.assertIn('charge_off_timing_distribution', patterns)
                self.assertIn('average_seasoning_at_charge_off', patterns)
        
        print("   ✓ Charge-off pattern analysis works correctly") 