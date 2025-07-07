# Notebooks

This folder contains interactive demonstrations and tutorials for the Vintage Forecasts system.

## Files

- **`vintage_forecasting_demo.qmd`**: Comprehensive Quarto demonstration of the vintage forecasting system
- **`styles.css`**: Custom styling for the Quarto document

## Vintage Forecasting Demo

The `vintage_forecasting_demo.qmd` file is a comprehensive Quarto document that demonstrates:

### Features Demonstrated

1. **Data Generation and Loading**
   - Synthetic vintage data generation
   - Complete vs. incomplete data formats
   - FICO band distribution analysis

2. **Flexible Data Handling**
   - Automatic data completion for incomplete vintage data
   - Comparison of data volume and processing efficiency
   - Validation of analysis accuracy

3. **Vintage Analysis by FICO Band**
   - FICO segmentation analysis
   - Performance metrics by credit quality
   - Seasoning curve fitting and visualization

4. **Forecasting with FICO Segmentation**
   - Multi-scenario forecasting (Conservative, Balanced, Aggressive)
   - Quality mix impact analysis
   - Monthly charge-off projections

5. **Risk Metrics and Analysis**
   - Volatility calculations
   - Portfolio concentration metrics
   - Risk comparison across scenarios

6. **Summary and Insights**
   - Key findings and implications
   - Recommendations for implementation
   - Next steps for development

### Rendering the Demo

To render the Quarto document to HTML:

```bash
# Install Quarto (if not already installed)
# Visit: https://quarto.org/docs/get-started/

# Render to HTML
quarto render vintage_forecasting_demo.qmd

# Or render with specific options
quarto render vintage_forecasting_demo.qmd --to html --toc
```

### Requirements

- **Quarto**: For rendering the document
- **Python**: For executing the code chunks
- **Required Python packages**: pandas, numpy, matplotlib, seaborn
- **System modules**: data_loader, vintage_analyzer, forecaster

### Output

The rendered HTML document includes:
- Interactive code execution
- Professional styling and formatting
- Table of contents navigation
- Responsive design for different screen sizes
- Print-friendly layout

### Customization

The demo can be customized by:
- Modifying the portfolio mix scenarios
- Adjusting the forecast horizon
- Changing the FICO band distributions
- Adding new analysis sections
- Customizing the visualizations

## Usage

1. **For Learning**: Run the demo to understand system capabilities
2. **For Development**: Use as a template for custom analyses
3. **For Presentations**: Render to HTML for stakeholder presentations
4. **For Documentation**: Include in system documentation

## Notes

- The demo uses synthetic data for demonstration purposes
- Real data can be substituted by modifying the data loading sections
- All visualizations are interactive and can be customized
- The document is self-contained and includes all necessary code 