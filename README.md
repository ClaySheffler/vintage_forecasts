# Vintage Forecasts: Quick Start

Vintage charge-off forecasting with FICO segmentation and flexible data handling.

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the main demo:**
   ```bash
   python main.py
   # or, for the interactive app:
   streamlit run notebooks/snowflake_streamlit.py
   ```

3. **Minimal code example:**
   ```python
   from src.data_loader import LoanDataLoader
   from src.vintage_analyzer import VintageAnalyzer
   from src.forecaster import ChargeOffForecaster

   # Load synthetic data
   data_loader = LoanDataLoader()
   loan_data = data_loader.load_data(source='synthetic')
   loan_data = data_loader.preprocess_data()

   # Analyze
   analyzer = VintageAnalyzer(loan_data)
   analyzer.calculate_vintage_metrics()
   print(analyzer.get_cumulative_gross_chargeoff_summary())
   ```

## Workflow Diagram

```mermaid
flowchart TD
    A[Input Data (CSV, Excel, or Synthetic)] --> B[LoanDataLoader]
    B --> C[VintageAnalyzer]
    C --> D[ChargeOffForecaster]
    D --> E[Outputs: Cumulative Gross Charge-off %, Plots, Reports]
    E --> F[Interactive Demo (Streamlit) or Notebooks]
```

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
├── src/                        # Core source code
│   ├── __init__.py
│   ├── data_loader.py          # Data loading and preprocessing with flexible format support
│   ├── vintage_analyzer.py     # Vintage analysis and seasoning curves by FICO band
│   └── forecaster.py           # Charge-off forecasting engine with FICO segmentation
├── docs/                       # Documentation
│   ├── SYSTEM_OVERVIEW.md      # Detailed system architecture documentation
│   └── FLEXIBLE_DATA_HANDLING.md # Comprehensive guide to flexible data handling
├── examples/                   # Example scripts and demonstrations
│   ├── example.py              # Step-by-step usage examples
│   └── demo_flexible_data.py   # Flexible data handling demonstration
├── tests/                      # Test suite
│   ├── test_vintage_forecasts.py   # Comprehensive test suite
│   ├── test_fico_segmentation.py   # FICO segmentation specific tests
│   └── test_system.py              # System integration tests
├── notebooks/                  # Jupyter notebooks
│   └── vintage_forecasting_demo.ipynb  # Interactive demo notebook
├── main.py                     # Main execution script with FICO segmentation demo
├── requirements.txt            # Python dependencies
├── LICENSE                     # MIT License
├── .gitignore                  # Git ignore rules
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
from src.data_loader import DataLoader
from src.vintage_analyzer import VintageAnalyzer
from src.forecaster import ChargeOffForecaster

# Load data with FICO segmentation
data_loader = DataLoader()
loan_data = data_loader.load_data(source='synthetic')  # or 'file' for real data
loan_data = data_loader.preprocess_data()

# Perform vintage analysis by FICO band
vintage_analyzer = VintageAnalyzer()
vintage_metrics = vintage_analyzer.analyze_vintage_data(loan_data)

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

### Available Scripts

#### Main Script (`main.py`)
Complete demonstration of the FICO-segmented vintage forecasting system:
```bash
python main.py
```

#### Example Script (`examples/example.py`)
Step-by-step examples of key features:
```bash
python examples/example.py
```

#### Flexible Data Handling Demo (`examples/demo_flexible_data.py`)
Demonstration of the flexible data handling feature:
```bash
python examples/demo_flexible_data.py
```

#### Test Suite
Run comprehensive tests:
```bash
python tests/test_vintage_forecasts.py
python tests/test_fico_segmentation.py
python tests/test_system.py
```

### Data Format Requirements

When using real data (source='file'), the system expects the following file format:

#### Required Columns

| Column Name | Data Type | Description | Example |
|-------------|-----------|-------------|---------|
| `loan_id` | String | Unique loan identifier | "LOAN_001234" |
| `vintage_date` | Date | Loan origination date | "2014-01-15" |
| `report_date` | Date | Performance reporting date | "2014-02-15" |
| `seasoning_month` | Integer | Months since origination | 1, 2, 3, ... |
| `fico_score` | Integer | FICO score at origination | 650, 720, 780, ... |
| `loan_amount` | Float | Original loan amount | 25000.00 |
| `charge_off_flag` | Integer | Binary charge-off flag (0=no charge-off, 1=charged off) | 0, 1 |
| `charge_off_amount` | Float | Amount charged off (0 if no charge-off, typically 70-90% of original principal if defaulted) | 0.00, 17500.00 |

#### Optional Columns

| Column Name | Data Type | Description | Example |
|-------------|-----------|-------------|---------|
| `outstanding_balance` | Float | Outstanding balance at report date | 24875.00 |
| `interest_rate` | Float | Loan interest rate | 0.085 (8.5%) |
| `term` | Integer | Loan term in months | 120 |
| `fico_band` | String | FICO score band | "650-699" |
| `risk_grade` | Integer | Risk grade (1-5) | 4 |

#### File Format Examples

**CSV Format:**
```csv
loan_id,vintage_date,report_date,seasoning_month,fico_score,loan_amount,charge_off_flag,charge_off_amount,outstanding_balance
LOAN_001,2014-01-15,2014-02-15,1,650,25000.00,0,0.00,24875.00
LOAN_001,2014-01-15,2014-03-15,2,650,25000.00,0,0.00,24800.37
LOAN_001,2014-01-15,2014-04-15,3,650,25000.00,1,17500.00,0.00
LOAN_002,2014-01-15,2014-02-15,1,720,35000.00,0,0.00,34947.50
```

**Excel Format:**
Same column structure as CSV, with multiple sheets supported.

**Parquet Format:**
Same column structure as CSV, with efficient compression.

#### Data Requirements

1. **Date Format**: Use ISO format (YYYY-MM-DD) or pandas-compatible date formats
2. **FICO Scores**: Must be integers between 300-850
3. **Charge-off Flags**: Must be 0 or 1 (binary flags indicating charge-off status)
4. **Monetary Amounts**: Use consistent currency units (e.g., USD)
5. **Seasoning Months**: Must be non-negative integers
6. **Data Completeness**: All required columns must be present
7. **Data Consistency**: Each loan should have records for all seasoning months from origination to current date

#### Loading Real Data

```python
# Load from CSV file
loan_data = data_loader.load_data(
    source='file',
    file_path='path/to/loan_data.csv',
    file_type='csv'
)

# Load from Excel file
loan_data = data_loader.load_data(
    source='file',
    file_path='path/to/loan_data.xlsx',
    file_type='excel'
)

# Load from Parquet file
loan_data = data_loader.load_data(
    source='file',
    file_path='path/to/loan_data.parquet',
    file_type='parquet'
)
```

### Interactive Demo

For an interactive demonstration with visualizations, run the Jupyter notebook:

```bash
jupyter notebook notebooks/vintage_forecasting_demo.ipynb
```

## Core Components

### 1. Data Loader (`src/data_loader.py`)

Handles loading and preprocessing of loan performance data with FICO segmentation and flexible data format support:

- **FICO Segmentation**: Assigns loans to FICO bands and risk grades
- **Flexible Data Handling**: Supports both complete and incomplete vintage data formats
- **Automatic Data Completion**: Fills missing seasoning months for charged-off loans
- **Synthetic Data Generation**: Creates realistic loan performance data with FICO characteristics
- **File Loading**: Supports CSV, Excel, and Parquet formats with FICO validation
- **Data Preprocessing**: Converts dates, calculates vintage metrics by FICO band, handles missing values
- **Quality Mix Analysis**: Tracks FICO band distribution and portfolio quality trends

### 2. Vintage Analyzer (`src/vintage_analyzer.py`)

Performs comprehensive vintage analysis with FICO segmentation and charge-off pattern analysis:

- **Vintage Metrics by FICO**: Calculates charge-off rates by vintage, seasoning, and FICO band
- **Seasoning Curves by FICO**: Fits mathematical curves to historical seasoning patterns for each FICO band
- **Charge-off Pattern Analysis**: Analyzes charge-off timing, amounts, and cumulative patterns by FICO band
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

Models how charge-off rates evolve over the life of loans for each FICO band using a multi-model approach:

- **Weibull Distribution**: Flexible, interpretable, commonly used for time-to-event data
- **Lognormal Distribution**: Captures right-skewed timing of losses, interpretable
- **Gompertz Curve**: Handles saturation and deceleration, interpretable
- **Simple Linear/Polynomial Trend**: Baseline, highly explainable, but may underfit
- **Scaling Method**: Scales observed charge-off rate at a given month by the reciprocal of the typical proportion of lifetime charge-offs observed at that month (simple financial scaling)
- **Additive Method**: Adds expected future charge-off rates (from historical averages) to the current observed rate (simple additive projection)
- **(Optional) Ensemble/Weighted Average**: Combines forecasts from above models, potentially improving robustness

> **Note:** All models are fit to the time series of cumulative charge-off rates by seasoning month. No macroeconomic factors are included in the base models unless such data is available and explicitly incorporated.

#### Model Assessment and Selection

For each model, the following are evaluated:
- **Goodness-of-fit**: $R^2$, RMSE, visual fit
- **Forecast stability**: Sensitivity to outliers, overfitting risk
- **Complexity**: Number of parameters, interpretability
- **Explainability**: Can the model's behavior be easily understood and justified?

A summary table is produced for each segment, showing the performance and characteristics of each model.

| Model         | RMSE   | R²     | # Params | Explainability | Notes                |
|---------------|--------|--------|----------|---------------|----------------------|
| Weibull CDF   | 0.012  | 0.98   | 2        | High          | Good fit, interpretable |
| Lognormal CDF | 0.013  | 0.97   | 2        | High          | Slightly underfits tail |
| Gompertz CDF  | 0.011  | 0.98   | 2        | High          | Best fit, similar to Weibull |
| Linear Trend  | 0.025  | 0.90   | 2        | Very High     | Underfits, but simple |
| Scaling       | 0.030  | 0.88   | 1        | Very High     | Simple financial scaling |
| Additive      | 0.028  | 0.89   | 1        | Very High     | Adds future expected CO% |
| Ensemble      | 0.011  | 0.98   | 4        | Medium        | Robust, less interpretable |

- If one model is clearly superior (accuracy, parsimony, and explainability), it is selected.
- If multiple models perform similarly, an ensemble or weighted average may be used to combine their forecasts, increasing robustness and trust.
- If models disagree significantly, this is flagged for further investigation and transparency in reporting.

### Future Work

- **Macroeconomic Integration**: Incorporate macroeconomic variables (e.g., unemployment, GDP, interest rates) if/when data is available, to allow for scenario-based or macro-driven forecasting.
- **Prepayment and Recovery Modeling**: Extend models to account for prepayments and recoveries, which can impact loss timing and magnitude.
- **Machine Learning Models**: Explore more complex models (e.g., gradient boosting, neural nets) if justified by data volume and need for accuracy, with careful attention to explainability.
- **Additional Features**: If data becomes available, consider incorporating borrower-level or loan-level features that may improve forecast accuracy, such as employment status, income, or geographic region.

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

## Documentation

### System Overview (`docs/SYSTEM_OVERVIEW.md`)
Comprehensive documentation of the system architecture, methodology, and implementation details.

### Flexible Data Handling (`docs/FLEXIBLE_DATA_HANDLING.md`)
Detailed guide to the flexible data handling feature, including usage examples and best practices.

### Interactive Demo (`examples/demo_flexible_data.py`)
Standalone demonstration script showing how the flexible data handling works.

## Support

For questions or support, please contact the development team or create an issue in the repository.

## Version History

- **v1.0.0**: Initial release with core forecasting functionality
- **v1.1.0**: Added scenario analysis and risk metrics
- **v1.2.0**: Enhanced visualizations and export capabilities
- **v1.3.0**: Added FICO segmentation and quality mix analysis
- **v1.4.0**: Added flexible data handling for complete and incomplete vintage data formats

## Data Handling Flexibility

The system is designed to handle both complete and incomplete vintage data formats:

### Complete Vintage Data (Traditional)
- Loans continue appearing in all seasoning months after charge-off
- Charge-off rate = 1 and charge-off amount = 0 for months after initial charge-off
- Outstanding balance = 0 for months after charge-off
- **Advantage**: Complete seasoning curves for all loans
- **Disadvantage**: Larger data volume, redundant records

### Incomplete Vintage Data (Alternative)
- Loans disappear from the data after charge-off
- Only the charge-off month record is included
- **Advantage**: Smaller data volume, cleaner data
- **Disadvantage**: Missing seasoning months for charged-off loans

### Automatic Data Completion
The system automatically detects and handles incomplete vintage data by:
1. Identifying charged-off loans that are missing from future seasoning months
2. Filling in missing seasoning months with appropriate charge-off flags
3. Propagating charge-off status (charge_off_flag = 1, charge_off_amount = 0, outstanding_balance = 0)
4. Ensuring complete seasoning curves for accurate vintage analysis

This allows you to use either data format without any changes to your analysis workflow.

### Example: Data Completion Process
```python
# Generate incomplete vintage data (loans disappear after charge-off)
incomplete_data = data_loader.generate_synthetic_data(
    num_vintages=3,
    loans_per_vintage=50,
    max_seasoning=24,
    incomplete_vintages=True  # Creates incomplete data
)

# Load and preprocess - automatically completes missing seasoning months
data_loader.data = incomplete_data
completed_data = data_loader.preprocess_data()

print(f"Original: {len(incomplete_data)} records")
print(f"Completed: {len(completed_data)} records")
# Both approaches produce identical analysis-ready data!
```

- At the individual loan/seasoning month level, use 'charge_off_flag' (0 or 1). 'charge_off_rate' is only meaningful for aggregated data (e.g., a group of loans).
