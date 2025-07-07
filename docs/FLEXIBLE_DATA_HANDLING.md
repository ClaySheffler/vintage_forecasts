# Flexible Data Handling in Vintage Forecasts

## Overview

The Vintage Forecasts system now supports flexible data handling that can automatically work with both complete and incomplete vintage data formats. This feature eliminates the need to manually format your data and ensures accurate analysis regardless of your data structure.

## Data Format Options

### Option 1: Complete Vintage Data (Traditional)
**Characteristics:**
- Loans continue appearing in all seasoning months after charge-off
- Charge-off flag = 1 and charge-off amount = 0 for months after initial charge-off
- Outstanding balance = 0 for months after charge-off

**Example:**
```
loan_id, vintage_date, seasoning_month, charge_off_flag, charge_off_amount, outstanding_balance
L001, 2020-01-01, 1, 0, 0.0, 35000.0
L001, 2020-01-01, 2, 0, 0.0, 34000.0
L001, 2020-01-01, 3, 1, 28000.0, 0.0      # Charge-off month
L001, 2020-01-01, 4, 1, 0.0, 0.0          # Continues after charge-off
L001, 2020-01-01, 5, 1, 0.0, 0.0          # Continues after charge-off
```

**Advantages:**
- Complete seasoning curves for all loans
- Traditional format used by many systems

**Disadvantages:**
- Larger data volume
- Redundant records after charge-off

### Option 2: Incomplete Vintage Data (Alternative)
**Characteristics:**
- Loans disappear from the data after charge-off
- Only the charge-off month record is included
- Cleaner, more compact data structure

**Example:**
```
loan_id, vintage_date, seasoning_month, charge_off_flag, charge_off_amount, outstanding_balance
L001, 2020-01-01, 1, 0, 0.0, 35000.0
L001, 2020-01-01, 2, 0, 0.0, 34000.0
L001, 2020-01-01, 3, 1, 28000.0, 0.0      # Charge-off month - last record
```

**Advantages:**
- Smaller data volume
- Cleaner data structure
- Easier to generate from operational systems

**Disadvantages:**
- Missing seasoning months for charged-off loans
- Requires completion for accurate vintage analysis

## Automatic Data Completion

The system automatically detects and handles incomplete vintage data through the `_complete_vintage_data()` method:

### How It Works

1. **Detection**: Identifies charged-off loans that are missing from future seasoning months
2. **Completion**: Fills in missing seasoning months with appropriate charge-off flags
3. **Propagation**: Sets charge_off_flag = 1, charge_off_amount = 0, outstanding_balance = 0 for future months
4. **Validation**: Ensures complete seasoning curves for accurate vintage analysis

### Example Completion Process

**Input (Incomplete):**
```
L001: seasoning_months = [1, 2, 3], charge_off_month = 3
L002: seasoning_months = [1, 2, 3, 4, 5], charge_off_month = 5
```

**Output (Completed):**
```
L001: seasoning_months = [1, 2, 3, 4, 5], charge_off_months = [3, 4, 5]
L002: seasoning_months = [1, 2, 3, 4, 5], charge_off_months = [5]
```

## Usage Examples

### Generating Synthetic Data

```python
from data_loader import DataLoader

data_loader = DataLoader()

# Generate incomplete vintage data
incomplete_data = data_loader.generate_synthetic_data(
    num_vintages=12,
    loans_per_vintage=100,
    max_seasoning=36,
    incomplete_vintages=True  # Loans disappear after charge-off
)

# Generate complete vintage data
complete_data = data_loader.generate_synthetic_data(
    num_vintages=12,
    loans_per_vintage=100,
    max_seasoning=36,
    incomplete_vintages=False  # Loans continue after charge-off
)
```

### Loading and Processing Real Data

```python
# Load your data (either format)
data_loader.load_data('your_data.csv')

# Preprocess automatically handles both formats
processed_data = data_loader.preprocess_data()

# Continue with analysis - no changes needed!
analyzer = VintageAnalyzer()
results = analyzer.analyze_vintage_data(processed_data)
```

### Working with Real Data Files

The system automatically detects your data format and handles it appropriately:

```python
# Your data can be in either format - the system handles it automatically
data_loader.load_data('incomplete_vintage_data.csv')  # Loans disappear after charge-off
# OR
data_loader.load_data('complete_vintage_data.csv')    # Loans continue after charge-off

# Same preprocessing call works for both
processed_data = data_loader.preprocess_data()
```

## Benefits

### For Data Providers
- **Flexibility**: Use whichever format is easier to generate
- **Efficiency**: Incomplete format reduces data volume by 30-50%
- **Simplicity**: No need to manually complete seasoning curves

### For Analysts
- **Transparency**: Automatic completion process is visible and auditable
- **Accuracy**: Ensures complete seasoning curves for accurate analysis
- **Consistency**: Same analysis workflow regardless of input format

### For Systems
- **Compatibility**: Works with existing data pipelines
- **Scalability**: Reduced data volume improves performance
- **Maintainability**: Single codebase handles both formats

## Data Volume Comparison

| Scenario | Complete Format | Incomplete Format | Volume Reduction |
|----------|----------------|-------------------|------------------|
| 12 vintages, 100 loans, 36 months | 43,200 records | ~28,800 records | ~33% |
| 24 vintages, 200 loans, 48 months | 230,400 records | ~153,600 records | ~33% |

*Note: Actual reduction depends on charge-off rates and timing*

## Implementation Details

### Key Methods

1. **`_complete_vintage_data()`**: Main completion logic
2. **`preprocess_data()`**: Entry point that calls completion
3. **`generate_synthetic_data()`**: Supports both formats via `incomplete_vintages` parameter

### Data Validation

The system validates completed data to ensure:
- All loans have complete seasoning curves
- Charge-off flags are properly propagated
- Outstanding balances are correctly set to zero after charge-off
- Report dates are calculated correctly

### Error Handling

- Gracefully handles edge cases (no charge-offs, missing data)
- Provides informative error messages
- Maintains data integrity throughout the process

## Migration Guide

### From Complete to Incomplete Format

If you want to switch to the more efficient incomplete format:

1. **Modify your data export** to stop including records after charge-off
2. **Update your data pipeline** to only export the charge-off month
3. **No changes needed** to your analysis code

### From Incomplete to Complete Format

If you prefer the traditional complete format:

1. **Modify your data export** to include all seasoning months
2. **Set charge_off_flag = 1** for months after charge-off
3. **Set charge_off_amount = 0** for months after charge-off
4. **Set outstanding_balance = 0** for months after charge-off

## Best Practices

### For Data Generation
- Use incomplete format for new data pipelines
- Include all required fields in your export
- Ensure charge-off amounts are captured in the charge-off month

### For Analysis
- Always use `preprocess_data()` before analysis
- Review completion logs for data quality insights
- Validate results against expected charge-off patterns

### For Production Systems
- Monitor data completion statistics
- Set up alerts for unusual completion patterns
- Document your chosen data format for team consistency

## Troubleshooting

### Common Issues

1. **Missing seasoning months**: System automatically completes them
2. **Inconsistent charge-off flags**: System validates and corrects
3. **Data volume concerns**: Use incomplete format for efficiency

### Debugging

Enable verbose logging to see completion details:

```python
# The system prints completion information automatically
data_loader.preprocess_data()
# Output: "Completing vintage data (filling missing seasoning months for charged-off loans)..."
# Output: "Completed vintage data: X records"
```

## Conclusion

The flexible data handling feature makes the Vintage Forecasts system more user-friendly and efficient. You can now use whichever data format works best for your organization without worrying about compatibility or accuracy issues. The automatic completion process ensures that your analysis is always based on complete, accurate vintage data. 