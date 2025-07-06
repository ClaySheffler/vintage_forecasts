# Vintage Charge-off Forecasting System

A comprehensive system for forecasting loan charge-offs using vintage analysis and time series methods. This project analyzes historical loan performance data from 2014 onwards and provides forward-looking projections for credit risk management.

## Overview

The system combines vintage analysis (loan performance by origination period) with seasoning curves (how charge-off rates evolve over loan life) to create accurate forecasts for future charge-offs. This is particularly valuable for long-term loans with terms exceeding 10 years.

## Key Features

- **Vintage Analysis**: Analyzes loan performance patterns by origination period
- **Seasoning Curves**: Models how charge-off rates evolve over the life of loans
- **Multi-Scenario Forecasting**: Generates forecasts under different economic conditions
- **Risk Metrics**: Calculates key risk indicators and concentration measures
- **Interactive Visualizations**: Comprehensive dashboards and charts
- **Export Capabilities**: Excel, CSV, and Parquet output formats

## Project Structure

```
vintage_forecasts/
├── src/
│   ├── __init__.py
│   ├── data_loader.py          # Data loading and preprocessing
│   ├── vintage_analyzer.py     # Vintage analysis and seasoning curves
│   └── forecaster.py           # Charge-off forecasting engine
├── notebooks/
│   └── vintage_forecasting_demo.ipynb  # Interactive demo notebook
├── outputs/                    # Generated reports and visualizations
├── main.py                     # Main execution script
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd vintage_forecasts
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the main script:
```bash
python main.py
```

## Usage

### Quick Start

Run the complete forecasting workflow:

```python
from src.data_loader import LoanDataLoader
from src.vintage_analyzer import VintageAnalyzer
from src.forecaster import ChargeOffForecaster

# Load data
data_loader = LoanDataLoader()
loan_data = data_loader.load_sample_data()
loan_data = data_loader.preprocess_data()

# Perform vintage analysis
vintage_analyzer = VintageAnalyzer(loan_data)
vintage_metrics = vintage_analyzer.calculate_vintage_metrics()
seasoning_curves = vintage_analyzer.fit_seasoning_curves()

# Create forecasts
forecaster = ChargeOffForecaster(vintage_analyzer, loan_data)
forecast = forecaster.forecast_portfolio_charge_offs(portfolio_data, forecast_end_date)
```

### Interactive Demo

For an interactive demonstration with visualizations, run the Jupyter notebook:

```bash
jupyter notebook notebooks/vintage_forecasting_demo.ipynb
```

## Core Components

### 1. Data Loader (`src/data_loader.py`)

Handles loading and preprocessing of loan performance data:

- **Sample Data Generation**: Creates realistic loan performance data for demonstration
- **File Loading**: Supports CSV, Excel, and Parquet formats
- **Data Preprocessing**: Converts dates, calculates vintage metrics, handles missing values

### 2. Vintage Analyzer (`src/vintage_analyzer.py`)

Performs comprehensive vintage analysis:

- **Vintage Metrics**: Calculates charge-off rates by vintage and seasoning
- **Seasoning Curves**: Fits mathematical curves to historical seasoning patterns
- **Vintage Quality**: Analyzes early performance indicators
- **Pattern Recognition**: Identifies seasonal and trend patterns

### 3. Forecaster (`src/forecaster.py`)

Generates charge-off forecasts:

- **Vintage Forecasting**: Projects charge-offs for individual vintages
- **Portfolio Forecasting**: Aggregates forecasts across entire portfolio
- **Scenario Analysis**: Generates forecasts under different economic conditions
- **Risk Metrics**: Calculates key risk indicators

## Methodology

### Vintage Analysis

The system analyzes loan performance by "vintage" (origination period) to understand:

- How different origination periods perform over time
- Seasonal patterns in loan quality
- Economic cycle effects on vintage performance
- Early warning indicators for vintage quality

### Seasoning Curves

Models how charge-off rates evolve over the life of loans using:

- **Weibull Distribution**: Flexible curve for modeling seasoning patterns
- **Lognormal Distribution**: Alternative approach for seasoning modeling
- **Gompertz Curve**: S-shaped curve for gradual seasoning

### Forecasting Approach

1. **Historical Analysis**: Fit seasoning curves to historical data
2. **Vintage Quality Assessment**: Evaluate quality of new vintages
3. **Forward Projection**: Apply seasoning curves to future periods
4. **Scenario Generation**: Create multiple economic scenarios
5. **Risk Assessment**: Calculate key risk metrics

## Outputs

The system generates several types of outputs:

### 1. Forecast Reports

- **Base Forecast**: Expected charge-off projections
- **Scenario Forecasts**: Optimistic, pessimistic, and stress scenarios
- **Risk Metrics**: Peak rates, timing, concentration measures

### 2. Visualizations

- **Vintage Analysis Dashboard**: Heatmaps, seasoning curves, quality metrics
- **Forecast Dashboard**: Monthly amounts, rates, cumulative totals
- **Scenario Comparison**: Side-by-side scenario analysis
- **Sensitivity Analysis**: Impact of parameter changes

### 3. Data Exports

- **Excel Files**: Comprehensive reports with multiple sheets
- **CSV Files**: Raw forecast data for further analysis
- **Parquet Files**: Efficient data storage for large datasets

## Key Metrics

### Portfolio Metrics

- **Total Charge-offs**: Sum of all projected charge-offs
- **Lifetime Loss Rate**: Total charge-offs as percentage of portfolio
- **Peak Charge-off Rate**: Highest monthly charge-off rate
- **Average Charge-off Rate**: Mean monthly charge-off rate

### Risk Metrics

- **Charge-off Volatility**: Standard deviation of monthly rates
- **Peak Timing**: When charge-offs are expected to peak
- **Vintage Concentration**: Herfindahl-Hirschman Index for vintage concentration
- **Scenario Sensitivity**: Range of outcomes across scenarios

## Use Cases

### Credit Risk Management

- **Capital Planning**: Estimate required capital reserves
- **Stress Testing**: Evaluate portfolio resilience under adverse conditions
- **Risk Appetite**: Set appropriate risk limits and thresholds

### Portfolio Management

- **Origination Strategy**: Optimize loan origination timing and volume
- **Pricing Strategy**: Adjust pricing based on expected losses
- **Portfolio Optimization**: Balance risk and return across vintages

### Regulatory Compliance

- **CECL Implementation**: Support Current Expected Credit Loss calculations
- **Stress Testing**: Meet regulatory stress testing requirements
- **Risk Reporting**: Provide comprehensive risk metrics for reporting

## Model Validation

The system includes several validation approaches:

1. **Historical Backtesting**: Compare forecasts to actual outcomes
2. **Out-of-Sample Testing**: Validate on held-out data
3. **Scenario Testing**: Evaluate performance under different conditions
4. **Sensitivity Analysis**: Test model robustness to parameter changes

## Limitations and Considerations

### Data Requirements

- **Historical Data**: Minimum 3-5 years of loan performance data
- **Vintage Granularity**: Monthly or quarterly vintage periods
- **Performance Metrics**: Charge-off rates, outstanding balances, loan characteristics

### Model Assumptions

- **Seasoning Patterns**: Assumes consistent seasoning behavior
- **Economic Stability**: May not capture structural changes
- **Vintage Quality**: Relies on historical patterns for new vintages

### Risk Factors

- **Model Risk**: Forecasts are estimates, not guarantees
- **Data Quality**: Results depend on accuracy of input data
- **Economic Changes**: Unforeseen economic events may impact forecasts

## Contributing

To contribute to this project:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For questions or support, please contact the development team or create an issue in the repository.

## Version History

- **v1.0.0**: Initial release with core forecasting functionality
- **v1.1.0**: Added scenario analysis and risk metrics
- **v1.2.0**: Enhanced visualizations and export capabilities
