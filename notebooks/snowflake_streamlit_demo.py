import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.vintage_analyzer import VintageAnalyzer
from src.forecaster import ChargeOffForecaster
import snowflake.connector

st.set_page_config(page_title="Vintage Forecasts: Snowflake Streamlit Demo", layout="wide")
st.title("Vintage Charge-off Forecasting: Snowflake Streamlit Demo")
st.markdown("""
This notebook demonstrates the full workflow of the Vintage Forecasts system using data loaded from a Snowflake database. 

**Instructions:**
- Edit the template SQL query below to match your own Snowflake schema and table names.
- Run each section in order to load, analyze, and forecast your loan portfolio.
- Visualizations are provided for data, forecasts, and model performance metrics.
""")

# --- 1. Snowflake Connection and Data Query ---
st.header("1. Connect to Snowflake and Load Data")

st.markdown("""
**Template Query:**
Edit this query to match your own database. The query should return a table with at least the following columns:
- `loan_id`, `vintage_date`, `seasoning_month`, `fico_score`, `loan_amount`, `charge_off_rate`, `charge_off_amount`, `outstanding_balance`, `term`

You may add or map additional columns as needed.
""")

def def_template_query():
    return '''
SELECT
    loan_id,
    vintage_date,
    seasoning_month,
    fico_score,
    loan_amount,
    charge_off_rate,
    charge_off_amount,
    outstanding_balance,
    term
FROM your_database.your_schema.your_loan_performance_table
WHERE vintage_date >= '2018-01-01'
'''

query = st.text_area("Edit your SQL query here:", value=def_template_query(), height=200)

with st.expander("Snowflake Connection Settings"):
    sf_account = st.text_input("Snowflake Account (e.g. xy12345.us-east-1)")
    sf_user = st.text_input("Username")
    sf_password = st.text_input("Password", type="password")
    sf_warehouse = st.text_input("Warehouse")
    sf_database = st.text_input("Database")
    sf_schema = st.text_input("Schema")
    connect_btn = st.button("Connect and Load Data")

@st.cache_data(show_spinner=True)
def load_data_from_snowflake(query, account, user, password, warehouse, database, schema):
    ctx = snowflake.connector.connect(
        account=account,
        user=user,
        password=password,
        warehouse=warehouse,
        database=database,
        schema=schema
    )
    df = pd.read_sql(query, ctx)
    ctx.close()
    return df

if connect_btn:
    with st.spinner("Loading data from Snowflake..."):
        try:
            df = load_data_from_snowflake(query, sf_account, sf_user, sf_password, sf_warehouse, sf_database, sf_schema)
            st.success(f"Loaded {len(df):,} records.")
            st.dataframe(df.head(100))
        except Exception as e:
            st.error(f"Error loading data: {e}")
else:
    df = None

# --- 2. Data Preprocessing ---
st.header("2. Data Preprocessing and FICO Band Assignment")

if df is not None:
    st.markdown("Assigning FICO bands and basic cleaning...")
    def assign_fico_band(fico):
        if fico >= 800: return '800+'
        elif fico >= 750: return '750-799'
        elif fico >= 700: return '700-749'
        elif fico >= 650: return '650-699'
        elif fico >= 600: return '600-649'
        else: return '<600'
    df['fico_band'] = df['fico_score'].apply(assign_fico_band)
    # Basic cleaning: drop rows with missing critical fields
    df = df.dropna(subset=['loan_id', 'vintage_date', 'seasoning_month', 'fico_score', 'loan_amount'])
    st.write("Sample after preprocessing:")
    st.dataframe(df.head(20))
    st.write("FICO Band Distribution:")
    st.bar_chart(df['fico_band'].value_counts())
else:
    st.info("Load data to continue.")

# --- 3. Vintage Analysis ---
st.header("3. Vintage Analysis and Seasoning Curves")

if df is not None:
    analyzer = VintageAnalyzer(df)
    st.markdown("Calculating vintage metrics and fitting seasoning curves by FICO band...")
    metrics = analyzer.calculate_vintage_metrics()
    curves = analyzer.fit_seasoning_curves()
    st.write("Vintage metrics sample:")
    st.dataframe(metrics.head(20))
    st.write("Fitted seasoning curves (R²):")
    r2_table = {band: {k: v['r_squared'] if v else None for k, v in curves[band].items()} for band in analyzer.fico_bands if band in curves}
    st.dataframe(pd.DataFrame(r2_table).T)
    # Visualize seasoning curves
    st.subheader("Seasoning Curves by FICO Band")
    fig, ax = plt.subplots(figsize=(10, 6))
    for band in analyzer.fico_bands:
        if band in curves and 'weibull' in curves[band] and curves[band]['weibull']:
            months = np.arange(1, 61)
            y = curves[band]['weibull']['function'](months, *curves[band]['weibull']['params'])
            ax.plot(months, y, label=f"{band} (Weibull)")
    ax.set_xlabel("Seasoning Month")
    ax.set_ylabel("Cumulative Charge-off Rate")
    ax.legend()
    st.pyplot(fig)

# --- 4. Forecasting ---
st.header("4. Forecasting Future Charge-offs")

if df is not None:
    st.markdown("Set forecast parameters and run the forecaster.")
    forecast_horizon = st.slider("Forecast Horizon (months)", min_value=12, max_value=120, value=60, step=12)
    # Portfolio mix by FICO band (dollar-weighted)
    st.write("Specify portfolio mix (dollar-weighted) by FICO band:")
    bands = sorted(df['fico_band'].unique())
    total_amt = float(df['loan_amount'].sum())
    mix = {}
    for band in bands:
        default = float(df[df['fico_band'] == band]['loan_amount'].sum()) / total_amt if total_amt > 0 else 1.0/len(bands)
        mix[band] = st.slider(f"{band}", min_value=0.0, max_value=1.0, value=round(default,2), step=0.01)
    # Normalize
    total_mix = sum(mix.values())
    if total_mix > 0:
        for band in mix:
            mix[band] /= total_mix
    st.write("Portfolio Mix:", mix)
    # Prepare portfolio mix dict for forecaster
    portfolio_mix = {band: {'loan_amount': mix[band]*total_amt, 'num_loans': int(mix[band]*len(df[df['fico_band']==band]))} for band in bands}
    forecaster = ChargeOffForecaster(analyzer, df)
    import datetime
    vintage_date = pd.to_datetime(df['vintage_date'].min())
    forecast = forecaster.forecast_vintage_charge_offs_by_fico(
        vintage_date=vintage_date,
        portfolio_mix=portfolio_mix,
        forecast_horizon=forecast_horizon
    )
    st.write("Forecast sample:")
    st.dataframe(forecast.head(20))
    # Visualize forecast
    st.subheader("Forecasted Cumulative Charge-off Rate")
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    for band in bands:
        band_forecast = forecast[forecast['fico_band'] == band]
        ax2.plot(band_forecast['seasoning_month'], band_forecast['cumulative_charge_off_rate'], label=band)
    ax2.set_xlabel("Seasoning Month")
    ax2.set_ylabel("Cumulative Charge-off Rate")
    ax2.legend()
    st.pyplot(fig2)

# --- 5. Model Performance Metrics ---
st.header("5. Model Performance Metrics and Comparison")

if df is not None:
    st.markdown("Model performance metrics (R², RMSE) for each FICO band and model:")
    perf_table = []
    for band in analyzer.fico_bands:
        if band in curves:
            for model_name, model_info in curves[band].items():
                if model_info:
                    perf_table.append({
                        'FICO Band': band,
                        'Model': model_name,
                        'R²': model_info['r_squared'],
                        # RMSE could be added if available in model_info
                    })
    st.dataframe(pd.DataFrame(perf_table))
    st.markdown("Compare model performance visually:")
    # Example: plot Weibull vs Lognormal for a selected band
    selected_band = st.selectbox("Select FICO band for model comparison:", bands)
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    months = np.arange(1, 61)
    for model_name in ['weibull', 'lognormal', 'gompertz']:
        if selected_band in curves and model_name in curves[selected_band] and curves[selected_band][model_name]:
            y = curves[selected_band][model_name]['function'](months, *curves[selected_band][model_name]['params'])
            ax3.plot(months, y, label=model_name.capitalize())
    ax3.set_xlabel("Seasoning Month")
    ax3.set_ylabel("Cumulative Charge-off Rate")
    ax3.legend()
    st.pyplot(fig3)

st.success("Demo complete. You can now explore your own data and forecasts interactively!") 