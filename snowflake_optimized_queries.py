"""
Optimized Snowflake SQL queries for vintage forecasting with large datasets (10M+ rows).

This approach moves heavy calculations into Snowflake SQL to minimize data transfer
and RAM usage in Python.
"""

def get_optimized_vintage_metrics_query(base_query: str) -> str:
    """
    Generate optimized SQL query that calculates vintage metrics in Snowflake
    instead of loading all data into pandas.
    
    This reduces data transfer from 10M+ rows to aggregated vintage metrics.
    """
    return f"""
    WITH cleaned_data AS (
        SELECT
            CAST(loan_id AS STRING) AS loan_id,
            vintage_date,
            seasoning_month,
            fico_score,
            loan_amount,
            charge_off_amount,
            outstanding_balance
        FROM ({base_query})
        WHERE 
            loan_id IS NOT NULL 
            AND TRIM(CAST(loan_id AS STRING)) != ''
            AND TRIM(CAST(loan_id AS STRING)) != 'nan'
            AND TRIM(CAST(loan_id AS STRING)) != 'None'
    ),
    fico_bands AS (
        SELECT 
            loan_id,
            vintage_date,
            seasoning_month,
            loan_amount,
            charge_off_amount,
            outstanding_balance,
            CASE 
                WHEN fico_score >= 800 THEN '800+'
                WHEN fico_score >= 750 THEN '750-799'
                WHEN fico_score >= 700 THEN '700-749'
                WHEN fico_score >= 650 THEN '650-699'
                WHEN fico_score >= 600 THEN '600-649'
                ELSE '<600'
            END AS fico_band
        FROM cleaned_data
    ),
    vintage_aggregates AS (
        SELECT 
            vintage_date,
            fico_band,
            seasoning_month,
            SUM(loan_amount) AS total_loan_amount,
            SUM(outstanding_balance) AS total_outstanding_balance,
            SUM(charge_off_amount) AS total_charge_off_amount,
            COUNT(DISTINCT loan_id) AS loan_count
        FROM fico_bands
        GROUP BY vintage_date, fico_band, seasoning_month
    ),
    vintage_metrics AS (
        SELECT 
            *,
            CASE 
                WHEN total_outstanding_balance > 0 
                THEN total_charge_off_amount / total_outstanding_balance 
                ELSE 0 
            END AS charge_off_flag,
            SUM(total_charge_off_amount) OVER (
                PARTITION BY vintage_date, fico_band 
                ORDER BY seasoning_month 
                ROWS UNBOUNDED PRECEDING
            ) AS cumulative_charge_off_amount,
            FIRST_VALUE(total_loan_amount) OVER (
                PARTITION BY vintage_date, fico_band 
                ORDER BY seasoning_month
            ) AS initial_loan_amount
        FROM vintage_aggregates
    )
    SELECT 
        vintage_date,
        fico_band,
        seasoning_month,
        total_loan_amount,
        total_outstanding_balance,
        total_charge_off_amount,
        loan_count,
        charge_off_flag,
        cumulative_charge_off_amount,
        initial_loan_amount,
        CASE 
            WHEN initial_loan_amount > 0 
            THEN cumulative_charge_off_amount / initial_loan_amount 
            ELSE 0 
        END AS cumulative_charge_off_flag
    FROM vintage_metrics
    ORDER BY vintage_date, fico_band, seasoning_month
    """

def get_optimized_loan_summary_query(base_query: str) -> str:
    """
    Generate SQL query to get loan-level summary for flexible data handling.
    This reduces the need to load all individual records.
    """
    return f"""
    WITH cleaned_data AS (
        SELECT
            CAST(loan_id AS STRING) AS loan_id,
            vintage_date,
            seasoning_month,
            fico_score,
            loan_amount,
            charge_off_amount,
            outstanding_balance
        FROM ({base_query})
        WHERE 
            loan_id IS NOT NULL 
            AND TRIM(CAST(loan_id AS STRING)) != ''
    ),
    loan_summary AS (
        SELECT 
            loan_id,
            vintage_date,
            MAX(seasoning_month) AS max_seasoning_month,
            MIN(seasoning_month) AS min_seasoning_month,
            SUM(loan_amount) AS total_loan_amount,
            SUM(charge_off_amount) AS total_charge_off_amount,
            MAX(outstanding_balance) AS final_outstanding_balance,
            CASE 
                WHEN fico_score >= 800 THEN '800+'
                WHEN fico_score >= 750 THEN '750-799'
                WHEN fico_score >= 700 THEN '700-749'
                WHEN fico_score >= 650 THEN '650-699'
                WHEN fico_score >= 600 THEN '600-649'
                ELSE '<600'
            END AS fico_band
        FROM cleaned_data
        GROUP BY loan_id, vintage_date, fico_score
    )
    SELECT * FROM loan_summary
    ORDER BY vintage_date, fico_band, loan_id
    """

def get_optimized_mature_vintage_query(base_query: str, mature_months: int = 72) -> str:
    """
    Generate SQL query to get mature vintage performance directly from Snowflake.
    """
    return f"""
    WITH cleaned_data AS (
        SELECT
            CAST(loan_id AS STRING) AS loan_id,
            vintage_date,
            seasoning_month,
            fico_score,
            loan_amount,
            charge_off_amount,
            outstanding_balance
        FROM ({base_query})
        WHERE 
            loan_id IS NOT NULL 
            AND TRIM(CAST(loan_id AS STRING)) != ''
    ),
    fico_bands AS (
        SELECT 
            *,
            CASE 
                WHEN fico_score >= 800 THEN '800+'
                WHEN fico_score >= 750 THEN '750-799'
                WHEN fico_score >= 700 THEN '700-749'
                WHEN fico_score >= 650 THEN '650-699'
                WHEN fico_score >= 600 THEN '600-649'
                ELSE '<600'
            END AS fico_band
        FROM cleaned_data
    ),
    vintage_aggregates AS (
        SELECT 
            vintage_date,
            fico_band,
            seasoning_month,
            SUM(loan_amount) AS total_loan_amount,
            SUM(charge_off_amount) AS total_charge_off_amount
        FROM fico_bands
        GROUP BY vintage_date, fico_band, seasoning_month
    ),
    mature_performance AS (
        SELECT 
            vintage_date,
            fico_band,
            SUM(total_charge_off_amount) AS total_charge_off_amount,
            SUM(total_loan_amount) AS total_loan_amount,
            CASE 
                WHEN SUM(total_loan_amount) > 0 
                THEN SUM(total_charge_off_amount) / SUM(total_loan_amount)
                ELSE 0 
            END AS cumulative_gross_chargeoff_pct
        FROM vintage_aggregates
        WHERE seasoning_month <= {mature_months}
        GROUP BY vintage_date, fico_band
    )
    SELECT * FROM mature_performance
    ORDER BY vintage_date, fico_band
    """

def get_optimized_forecast_data_query(base_query: str, focus_years: int = 5) -> str:
    """
    Generate SQL query to get data needed for forecasting (recent vintages only).
    """
    return f"""
    WITH cleaned_data AS (
        SELECT
            CAST(loan_id AS STRING) AS loan_id,
            vintage_date,
            seasoning_month,
            fico_score,
            loan_amount,
            charge_off_amount,
            outstanding_balance
        FROM ({base_query})
        WHERE 
            loan_id IS NOT NULL 
            AND TRIM(CAST(loan_id AS STRING)) != ''
            AND vintage_date >= DATEADD(year, -{focus_years}, CURRENT_DATE())
    ),
    fico_bands AS (
        SELECT 
            *,
            CASE 
                WHEN fico_score >= 800 THEN '800+'
                WHEN fico_score >= 750 THEN '750-799'
                WHEN fico_score >= 700 THEN '700-749'
                WHEN fico_score >= 650 THEN '650-699'
                WHEN fico_score >= 600 THEN '600-649'
                ELSE '<600'
            END AS fico_band
        FROM cleaned_data
    ),
    vintage_aggregates AS (
        SELECT 
            vintage_date,
            fico_band,
            seasoning_month,
            SUM(loan_amount) AS total_loan_amount,
            SUM(charge_off_amount) AS total_charge_off_amount,
            SUM(outstanding_balance) AS total_outstanding_balance
        FROM fico_bands
        GROUP BY vintage_date, fico_band, seasoning_month
    ),
    cumulative_metrics AS (
        SELECT 
            *,
            SUM(total_charge_off_amount) OVER (
                PARTITION BY vintage_date, fico_band 
                ORDER BY seasoning_month 
                ROWS UNBOUNDED PRECEDING
            ) AS cumulative_charge_off_amount,
            FIRST_VALUE(total_loan_amount) OVER (
                PARTITION BY vintage_date, fico_band 
                ORDER BY seasoning_month
            ) AS initial_loan_amount
        FROM vintage_aggregates
    )
    SELECT 
        vintage_date,
        fico_band,
        seasoning_month,
        total_loan_amount,
        total_outstanding_balance,
        total_charge_off_amount,
        cumulative_charge_off_amount,
        initial_loan_amount,
        CASE 
            WHEN initial_loan_amount > 0 
            THEN cumulative_charge_off_amount / initial_loan_amount 
            ELSE 0 
        END AS cumulative_charge_off_flag
    FROM cumulative_metrics
    ORDER BY vintage_date, fico_band, seasoning_month
    """

def get_optimized_fico_mix_query(base_query: str) -> str:
    """
    Generate SQL query to get FICO band mix by vintage.
    """
    return f"""
    WITH cleaned_data AS (
        SELECT
            CAST(loan_id AS STRING) AS loan_id,
            vintage_date,
            fico_score,
            loan_amount
        FROM ({base_query})
        WHERE 
            loan_id IS NOT NULL 
            AND TRIM(CAST(loan_id AS STRING)) != ''
    ),
    fico_bands AS (
        SELECT 
            loan_id,
            vintage_date,
            loan_amount,
            CASE 
                WHEN fico_score >= 800 THEN '800+'
                WHEN fico_score >= 750 THEN '750-799'
                WHEN fico_score >= 700 THEN '700-749'
                WHEN fico_score >= 650 THEN '650-699'
                WHEN fico_score >= 600 THEN '600-649'
                ELSE '<600'
            END AS fico_band
        FROM cleaned_data
    ),
    vintage_mix AS (
        SELECT 
            vintage_date,
            fico_band,
            SUM(loan_amount) AS band_loan_amount,
            COUNT(DISTINCT loan_id) AS band_loan_count
        FROM fico_bands
        GROUP BY vintage_date, fico_band
    ),
    vintage_totals AS (
        SELECT 
            vintage_date,
            SUM(band_loan_amount) AS total_loan_amount,
            SUM(band_loan_count) AS total_loan_count
        FROM vintage_mix
        GROUP BY vintage_date
    )
    SELECT 
        vm.vintage_date,
        vm.fico_band,
        vm.band_loan_amount,
        vm.band_loan_count,
        vt.total_loan_amount,
        vt.total_loan_count,
        vm.band_loan_amount / vt.total_loan_amount AS amount_pct,
        vm.band_loan_count / vt.total_loan_count AS count_pct
    FROM vintage_mix vm
    JOIN vintage_totals vt ON vm.vintage_date = vt.vintage_date
    ORDER BY vm.vintage_date, vm.fico_band
    """

# Example usage function
def example_optimized_workflow():
    """
    Example of how to use optimized queries for large datasets
    """
    base_query = '''
    SELECT
        loan_id,
        vintage_date,
        report_date,
        seasoning_month,
        fico_score,
        loan_amount,
        charge_off_flag,
        charge_off_amount,
        outstanding_balance,
        term,
        interest_rate
    FROM your_database.your_schema.your_loan_performance_table
    WHERE vintage_date >= '2018-01-01'
    '''
    
    from snowflake.snowpark.context import get_active_session
    session = get_active_session()
    
    print("=== OPTIMIZED SNOWFLAKE WORKFLOW FOR LARGE DATASETS ===\n")
    
    # 1. Get vintage metrics (much smaller dataset)
    print("1. Loading vintage metrics (aggregated data)...")
    vintage_metrics_query = get_optimized_vintage_metrics_query(base_query)
    vintage_metrics = session.sql(vintage_metrics_query).to_pandas()
    print(f"   ✓ Loaded {len(vintage_metrics):,} vintage metric records (vs 10M+ raw records)")
    
    # 2. Get loan summary for flexible data handling
    print("\n2. Loading loan summary...")
    loan_summary_query = get_optimized_loan_summary_query(base_query)
    loan_summary = session.sql(loan_summary_query).to_pandas()
    print(f"   ✓ Loaded {len(loan_summary):,} loan summary records")
    
    # 3. Get mature vintage performance
    print("\n3. Loading mature vintage performance...")
    mature_query = get_optimized_mature_vintage_query(base_query)
    mature_performance = session.sql(mature_query).to_pandas()
    print(f"   ✓ Loaded {len(mature_performance):,} mature vintage records")
    
    # 4. Get forecast data (recent vintages only)
    print("\n4. Loading forecast data (recent vintages)...")
    forecast_query = get_optimized_forecast_data_query(base_query)
    forecast_data = session.sql(forecast_query).to_pandas()
    print(f"   ✓ Loaded {len(forecast_data):,} forecast data records")
    
    # 5. Get FICO mix
    print("\n5. Loading FICO mix data...")
    fico_mix_query = get_optimized_fico_mix_query(base_query)
    fico_mix = session.sql(fico_mix_query).to_pandas()
    print(f"   ✓ Loaded {len(fico_mix):,} FICO mix records")
    
    print(f"\n=== SUMMARY ===")
    print(f"Total records loaded: {len(vintage_metrics) + len(loan_summary) + len(mature_performance) + len(forecast_data) + len(fico_mix):,}")
    print(f"Memory usage: ~{sum([df.memory_usage(deep=True).sum() for df in [vintage_metrics, loan_summary, mature_performance, forecast_data, fico_mix]]) / 1024 / 1024:.1f} MB")
    print(f"vs 10M+ raw records that would require ~{10_000_000 * 100 / 1024 / 1024:.0f} MB")
    
    return {
        'vintage_metrics': vintage_metrics,
        'loan_summary': loan_summary,
        'mature_performance': mature_performance,
        'forecast_data': forecast_data,
        'fico_mix': fico_mix
    }

if __name__ == "__main__":
    example_optimized_workflow() 