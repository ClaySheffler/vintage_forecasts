# Vintage Charge-off Forecasting System with FICO Segmentation

A comprehensive system for forecasting loan charge-offs using vintage analysis and FICO score segmentation. This project analyzes historical loan performance data from 2014 onwards and provides forward-looking projections for credit risk management with quality mix analysis.

## Overview

The system combines vintage analysis (loan performance by origination period) with FICO segmentation and seasoning curves (how charge-off rates evolve over loan life) to create accurate forecasts for future charge-offs. This is particularly valuable for long-term loans with terms exceeding 10 years and portfolios with varying credit quality characteristics.

## FICO Segmentation

The system segments loans into FICO score bands and corresponding risk grades:

- **600-649**: Very High Risk (Grade 5)
- **650-699**: High Risk (Grade 4) 
- **700-749**: Medium Risk (Grade 3)
- **750-799**: Low Risk (Grade 2)
- **800+**: Very Low Risk (Grade 1)

Each FICO band has distinct seasoning patterns, vintage performance characteristics, and charge-off rate profiles.

## Key Features

- **FICO Segmentation**: Analyzes loan performance by FICO score bands and risk grades
- **Quality Mix Analysis**: Tracks and forecasts changes in portfolio quality composition
- **Vintage Analysis by FICO**: Analyzes loan performance patterns by origination period and FICO band
- **Seasoning Curves by FICO**: Models how charge-off rates evolve over loan life for each FICO band
- **Dollar-Weighted Aggregation**: Combines FICO band forecasts using dollar-weighted averages
- **Multi-Scenario Forecasting**: Generates forecasts under different economic and quality mix scenarios
- **Risk Metrics**: Calculates key risk indicators and concentration measures by FICO band and portfolio level
- **Interactive Visualizations**: Comprehensive dashboards and charts with FICO segmentation
- **Export Capabilities**: Excel, CSV, and Parquet output formats with FICO breakdowns

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

Run the complete FICO-segmented forecasting workflow:

```python
from src.data_loader import LoanDataLoader
from src.vintage_analyzer import VintageAnalyzer
from src.forecaster import ChargeOffForecaster

# Load data with FICO segmentation
data_loader = LoanDataLoader()
loan_data = data_loader.load_data(source='synthetic')  # or 'file' for real data
loan_data = data_loader.preprocess_data()

# Perform vintage analysis by FICO band
vintage_analyzer = VintageAnalyzer(loan_data)
vintage_metrics = vintage_analyzer.calculate_vintage_metrics()
seasoning_curves = vintage_analyzer.fit_seasoning_curves()

# Create FICO-segmented forecasts
forecaster = ChargeOffForecaster(vintage_analyzer, loan_data)

# Define portfolio mix by FICO band
portfolio_mix = {
    '600-649': {'loan_amount': 5000000, 'num_loans': 200},   # 5% Very High Risk
    '650-699': {'loan_amount': 15000000, 'num_loans': 600},  # 15% High Risk
    '700-749': {'loan_amount': 30000000, 'num_loans': 1200}, # 30% Medium Risk
    '750-799': {'loan_amount': 35000000, 'num_loans': 1400}, # 35% Low Risk
    '800+': {'loan_amount': 15000000, 'num_loans': 600}      # 15% Very Low Risk
}

# Forecast by FICO band and aggregate
forecast = forecaster.forecast_vintage_charge_offs_by_fico(
    vintage_date=datetime(2024, 1, 1),
    portfolio_mix=portfolio_mix,
    forecast_horizon=120
)
```

### Interactive Demo

For an interactive demonstration with visualizations, run the Jupyter notebook:

```bash
jupyter notebook notebooks/vintage_forecasting_demo.ipynb
```

## Core Components

### 1. Data Loader (`src/data_loader.py`)

Handles loading and preprocessing of loan performance data with FICO segmentation:

- **FICO Segmentation**: Assigns loans to FICO bands and risk grades
- **Synthetic Data Generation**: Creates realistic loan performance data with FICO characteristics
- **File Loading**: Supports CSV, Excel, and Parquet formats with FICO validation
- **Data Preprocessing**: Converts dates, calculates vintage metrics by FICO band, handles missing values
- **Quality Mix Analysis**: Tracks FICO band distribution and portfolio quality trends

### 2. Vintage Analyzer (`src/vintage_analyzer.py`)

Performs comprehensive vintage analysis with FICO segmentation:

- **Vintage Metrics by FICO**: Calculates charge-off rates by vintage, seasoning, and FICO band
- **Seasoning Curves by FICO**: Fits mathematical curves to historical seasoning patterns for each FICO band
- **Vintage Quality by FICO**: Analyzes early performance indicators by credit quality
- **Pattern Recognition**: Identifies seasonal and trend patterns by FICO band
- **Quality Mix Trends**: Analyzes changes in portfolio quality composition over time
- **Aggregate Analysis**: Provides dollar-weighted portfolio-level metrics

### 3. Forecaster (`src/forecaster.py`)

Generates charge-off forecasts with FICO segmentation:

- **FICO Band Forecasting**: Projects charge-offs for individual FICO bands within vintages
- **Dollar-Weighted Aggregation**: Combines FICO band forecasts using dollar-weighted averages
- **Portfolio Forecasting**: Aggregates forecasts across entire portfolio with quality mix analysis
- **Quality Mix Scenarios**: Generates forecasts under different portfolio quality compositions
- **Scenario Analysis**: Generates forecasts under different economic and quality mix conditions
- **Risk Metrics**: Calculates key risk indicators by FICO band and portfolio level

## Methodology

### FICO Segmentation and Vintage Analysis

The system analyzes loan performance by "vintage" (origination period) and FICO band to understand:

- How different origination periods perform over time by credit quality
- Seasonal patterns in loan quality across FICO bands
- Economic cycle effects on vintage performance by risk grade
- Early warning indicators for vintage quality by FICO band
- Portfolio quality mix trends and their impact on overall performance

### Seasoning Curves by FICO Band

Models how charge-off rates evolve over the life of loans for each FICO band using:

- **Weibull Distribution**: Flexible curve for modeling seasoning patterns by credit quality
- **Lognormal Distribution**: Alternative approach for seasoning modeling by FICO band
- **Gompertz Curve**: S-shaped curve for gradual seasoning by risk grade
- **FICO-Specific Parameters**: Each FICO band has distinct seasoning characteristics
- **Quality Mix Impact**: Aggregates seasoning curves using dollar-weighted averages

### Forecasting Approach with FICO Segmentation

1. **Historical Analysis by FICO**: Fit seasoning curves to historical data for each FICO band
2. **Vintage Quality Assessment by FICO**: Evaluate quality of new vintages by credit quality
3. **Quality Mix Analysis**: Assess current and projected portfolio quality composition
4. **Forward Projection by FICO**: Apply seasoning curves to future periods for each FICO band
5. **Dollar-Weighted Aggregation**: Combine FICO band forecasts using dollar-weighted averages
6. **Quality Mix Scenarios**: Create multiple quality mix scenarios (Conservative, Balanced, Aggressive)
7. **Economic Scenarios**: Create multiple economic scenarios for each quality mix
8. **Risk Assessment**: Calculate key risk metrics by FICO band and portfolio level

## Outputs

The system generates several types of outputs:

### 1. Forecast Reports

- **Base Forecast**: Expected charge-off projections with FICO band breakdowns
- **Quality Mix Scenarios**: Conservative, balanced, and aggressive quality mix forecasts
- **Economic Scenarios**: Optimistic, pessimistic, and stress scenarios for each quality mix
- **FICO Band Analysis**: Detailed charge-off projections by FICO band and risk grade
- **Risk Metrics**: Peak rates, timing, concentration measures by FICO band and portfolio level

### 2. Visualizations

- **FICO-Segmented Vintage Analysis Dashboard**: Heatmaps, seasoning curves, quality metrics by FICO band
- **Quality Mix Analysis**: Portfolio quality trends and FICO band distribution charts
- **Forecast Dashboard**: Monthly amounts, rates, cumulative totals with FICO breakdowns
- **Quality Mix Scenarios**: Side-by-side quality mix scenario analysis
- **Economic Scenarios**: Side-by-side economic scenario analysis for each quality mix
- **FICO Band Performance**: Individual FICO band seasoning curves and vintage performance
- **Sensitivity Analysis**: Impact of parameter changes by FICO band and quality mix

### 3. Data Exports

- **Excel Files**: Comprehensive reports with multiple sheets including FICO breakdowns
- **CSV Files**: Raw forecast data with FICO band analysis for further analysis
- **Parquet Files**: Efficient data storage for large datasets with FICO segmentation
- **FICO Mix Analysis**: Detailed quality mix trends and portfolio composition reports

## Key Metrics

### Portfolio Metrics

- **Total Charge-offs**: Sum of all projected charge-offs (aggregated from FICO bands)
- **Lifetime Loss Rate**: Total charge-offs as percentage of portfolio (dollar-weighted)
- **Peak Charge-off Rate**: Highest monthly charge-off rate (aggregate)
- **Average Charge-off Rate**: Mean monthly charge-off rate (aggregate)
- **Quality Mix Impact**: Difference in lifetime loss rates between quality mix scenarios
- **FICO Band Concentration**: Distribution of portfolio across FICO bands and risk grades

### Risk Metrics

- **Charge-off Volatility**: Standard deviation of monthly rates (by FICO band and aggregate)
- **Peak Timing**: When charge-offs are expected to peak (by FICO band and aggregate)
- **Vintage Concentration**: Herfindahl-Hirschman Index for vintage concentration
- **FICO Band Concentration**: Herfindahl-Hirschman Index for FICO band concentration
- **Quality Mix Sensitivity**: Range of outcomes across quality mix scenarios
- **Economic Scenario Sensitivity**: Range of outcomes across economic scenarios
- **FICO Band Risk Contribution**: Individual FICO band contribution to portfolio risk

## Use Cases

### Credit Risk Management

- **Capital Planning**: Estimate required capital reserves with FICO band breakdowns
- **Quality Mix Stress Testing**: Evaluate portfolio resilience under different quality mix scenarios
- **Economic Stress Testing**: Evaluate portfolio resilience under adverse economic conditions
- **Risk Appetite**: Set appropriate risk limits and thresholds by FICO band
- **Quality Mix Limits**: Establish limits on portfolio quality composition

### Portfolio Management

- **Quality Mix Strategy**: Optimize portfolio composition across FICO bands
- **Origination Strategy**: Optimize loan origination timing and volume by FICO band
- **Pricing Strategy**: Adjust pricing based on expected losses by credit quality
- **Portfolio Optimization**: Balance risk and return across vintages and FICO bands
- **Quality Mix Monitoring**: Track and manage portfolio quality composition trends

### Regulatory Compliance

- **CECL Implementation**: Support Current Expected Credit Loss calculations with FICO segmentation
- **Quality Mix Stress Testing**: Meet regulatory stress testing requirements including quality mix scenarios
- **Economic Stress Testing**: Meet regulatory stress testing requirements under adverse economic conditions
- **Risk Reporting**: Provide comprehensive risk metrics by FICO band for regulatory reporting
- **Portfolio Quality Reporting**: Track and report portfolio quality composition trends

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
