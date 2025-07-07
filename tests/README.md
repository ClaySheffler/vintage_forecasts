# Test Suite

This folder contains comprehensive tests for the Vintage Forecasts system.

## Files

- **`test_vintage_forecasts.py`**: Comprehensive test suite covering core functionality
- **`test_fico_segmentation.py`**: FICO segmentation specific tests
- **`test_system.py`**: System integration tests

## Purpose

These tests ensure:
- **Code Quality**: All functionality works as expected
- **Regression Prevention**: Changes don't break existing features
- **Feature Validation**: New features are properly implemented
- **Documentation**: Tests serve as usage examples
- **Confidence**: Reliable system for production use

## Test Coverage

### Core Functionality (`test_vintage_forecasts.py`)
- Data loading and preprocessing
- Vintage analysis and seasoning curves
- Forecasting and scenario analysis
- FICO segmentation and quality mix analysis
- Flexible data handling

### FICO Segmentation (`test_fico_segmentation.py`)
- FICO band assignment and validation
- Risk grade calculations
- Quality mix analysis
- FICO-specific seasoning patterns
- Dollar-weighted aggregation

### System Integration (`test_system.py`)
- End-to-end workflow testing
- Data flow validation
- Error handling and edge cases
- Performance and scalability
- Integration between components

## Usage

### Running Tests

```bash
# Run all tests
python tests/test_vintage_forecasts.py
python tests/test_fico_segmentation.py
python tests/test_system.py

# Run specific test file
python tests/test_vintage_forecasts.py

# Run with verbose output
python -v tests/test_vintage_forecasts.py
```

### Test Development

When adding new features:
1. Write tests first (TDD approach)
2. Ensure comprehensive coverage
3. Test edge cases and error conditions
4. Update this README if adding new test files

## Quality Assurance

- **Unit Tests**: Individual component testing
- **Integration Tests**: Component interaction testing
- **Regression Tests**: Ensure no breaking changes
- **Performance Tests**: Validate system performance
- **Edge Case Tests**: Handle unusual scenarios 