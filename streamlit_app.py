import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from scipy.stats import norm
import streamlit as st
import empyrical as ep

# Version session state 0.1

def load_data(uploaded_file):
    # Load the data from uploaded Excel file
    df_returns = pd.read_excel(uploaded_file, sheet_name='Returns', index_col=0, parse_dates=True)
    df_weights = pd.read_excel(uploaded_file, sheet_name='Weights', index_col=0, parse_dates=True)
    return df_returns, df_weights

def generate_rebalanced_weights_for_frequency(strategic_weights, rebalance_frequency, start_date, returns_df):
    start_date = pd.to_datetime(start_date)

    # Ensure strategic_weights is always a DataFrame for consistency
    if isinstance(strategic_weights, pd.Series):
        strategic_weights = strategic_weights.to_frame().T
    elif isinstance(strategic_weights, np.ndarray):
        # This part of the code should actually never be reached based on your setup,
        # since strategic_weights should be either a Series or DataFrame, but just in case
        # you directly passed an ndarray, handle it appropriately by converting to DataFrame.
        # This requires knowing the column names in advance, which should match df_returns columns or be predefined.
        column_names = strategic_weights.columns  # Assuming strategic weights align with returns columns
        strategic_weights = pd.DataFrame(strategic_weights, columns=column_names)

    # Find the closest start date in returns_df to the provided start_date
    closest_start_date = returns_df.index[min(range(len(returns_df.index)), key=lambda i: abs(returns_df.index[i] - start_date))]

    # Generate rebalance dates starting from the closest_start_date
    if rebalance_frequency == 'Daily':
        rebalance_dates = returns_df.loc[closest_start_date:].index
    elif rebalance_frequency == 'Weekly':
        rebalance_dates = returns_df.loc[closest_start_date:].resample('W-WED').last().dropna().index
    elif rebalance_frequency == 'Monthly':
        rebalance_dates = returns_df.loc[closest_start_date:].resample('M').last().dropna().index
    elif rebalance_frequency == 'Quarterly':
        rebalance_dates = returns_df.loc[closest_start_date:].resample('Q').last().dropna().index
    elif rebalance_frequency == 'Annually':
        rebalance_dates = returns_df.loc[closest_start_date:].resample('A').last().dropna().index
    elif rebalance_frequency == 'Buy and Hold':
        rebalance_dates = pd.DatetimeIndex([returns_df.index.min()])

    # Initialize a DataFrame to hold rebalanced weights with correct column names from strategic_weights
    rebalanced_weights = pd.DataFrame(index=rebalance_dates, columns=strategic_weights.columns, dtype=float)

    # Assign the strategic weights for each rebalance date
    for date in rebalance_dates:
        rebalanced_weights.loc[date] = strategic_weights.iloc[0].values  # Using .values to ensure proper assignment

    return rebalanced_weights

def adjust_rebalancing_dates(rebalancing_weights, df_returns):
    """
    Adjusts the rebalancing dates to align with df_returns index dates.
    Maps each rebalancing date to the closest next date in df_returns if not directly matched.
    """
    adjusted_rebalancing_weights = pd.DataFrame(columns=rebalancing_weights.columns)

    for date in rebalancing_weights.index:
        if date in df_returns.index:
            # If the date matches directly, use it
            adjusted_date = date
        else:
            # Find the closest next date in df_returns
            future_dates = df_returns.index[df_returns.index > date]
            if not future_dates.empty:
                adjusted_date = future_dates[0]
            else:
                # If no future dates are available, ignore this rebalancing weight
                continue

        # Map the weights to the adjusted date
        adjusted_rebalancing_weights.loc[adjusted_date] = rebalancing_weights.loc[date]

    return adjusted_rebalancing_weights

def calculate_detailed_portfolio(df_returns, rebalancing_weights, start_date, portfolio_name="Portfolio"):
    start_date = pd.to_datetime(start_date)
    df_returns = df_returns.loc[start_date:].copy()
    rebalancing_weights = rebalancing_weights.loc[start_date:].copy()
    # Assuming adjust_rebalancing_dates is a function that adjusts the dates
    rebalancing_weights = adjust_rebalancing_dates(rebalancing_weights, df_returns)

    assets = df_returns.columns
    df_portfolio = pd.DataFrame(index=df_returns.index,
                                columns=['Rebalancing', 'Portfolio Return', 'Portfolio Index'] + [f'BOD_W_{asset}' for
                                                                                                  asset in assets] + [
                                            f'EOD_W_{asset}' for asset in assets])
    df_portfolio.fillna(0, inplace=True)  # Initialize dataframe with zeros
    df_portfolio['Portfolio Index'] = 100.0  # Start with an index of 100

    # Initialize EOD weights with the first set of rebalancing weights
    eod_weights = rebalancing_weights.iloc[0].values if not rebalancing_weights.empty else np.zeros(len(assets))

    # Ensure the first day is treated as a rebalancing day
    first_rebalance_flag = True

    for i, date in enumerate(df_returns.index):
        is_rebalancing_day = date in rebalancing_weights.index or first_rebalance_flag
        df_portfolio.loc[date, 'Rebalancing'] = is_rebalancing_day  # Use boolean for rebalancing flag

        if is_rebalancing_day:
            # Apply rebalanced weights if it's a rebalancing day or the first day
            eod_weights = rebalancing_weights.loc[date].values if date in rebalancing_weights.index else eod_weights
            first_rebalance_flag = False  # Reset flag after first rebalance

        if i == 0:
            # Special treatment for the first day
            df_portfolio.loc[date, [f'BOD_W_{asset}' for asset in assets]] = eod_weights
        else:
            # For subsequent days, set BOD weights as the previous day's EOD weights
            previous_date = df_returns.index[i - 1]
            df_portfolio.loc[date, [f'BOD_W_{asset}' for asset in assets]] = df_portfolio.loc[
                previous_date, [f'EOD_W_{asset}' for asset in assets]].values

        # Calculate the portfolio return for the day using BOD weights
        bod_weights = df_portfolio.loc[date, [f'BOD_W_{asset}' for asset in assets]].values.astype(float)
        portfolio_return = np.dot(bod_weights, df_returns.loc[date].values)
        df_portfolio.loc[date, 'Portfolio Return'] = portfolio_return

        # Update portfolio index
        if i > 0:  # Skip index update for the first day
            prev_index = df_portfolio.loc[previous_date, 'Portfolio Index']
            df_portfolio.loc[date, 'Portfolio Index'] = prev_index * (1 + portfolio_return)

        # Non-rebalancing day: Float EOD weights based on daily returns
        if not df_portfolio.loc[date, 'Rebalancing']:
            daily_returns = df_returns.loc[date].values
            eod_weights *= (1 + daily_returns)
            eod_weights /= eod_weights.sum()  # Normalize to ensure total weight is 1

        # Set EOD weights for all days
        df_portfolio.loc[date, [f'EOD_W_{asset}' for asset in assets]] = eod_weights

    df_portfolio['Portfolio Name'] = portfolio_name

    return df_portfolio

def calculate_risk_metrics(df_portfolio, df_returns, start_date, window=30):
    # Calculate the exponentially weighted moving average (EWMA) covariance matrix
    df_returns_subset = df_returns.loc[start_date:]

    ewma_cov = df_returns.ewm(span=window).cov(pairwise=True)

    # Initialize the dataframe to store risk metrics
    df_risk = pd.DataFrame(index=df_returns_subset.index)

    # Extract begin-of-day weights from df_portfolio
    bod_weights_df = df_portfolio[[col for col in df_portfolio.columns if 'BOD_W_' in col]]
    # Fix zero weights at T0
    bod_weights_df.iloc[0] = bod_weights_df.iloc[1]

    bod_weights_df.columns = [col.replace('BOD_W_', '') for col in bod_weights_df.columns]

    for date in df_returns_subset.index:
        # Extract the begin-of-day weights for the current date
        if date in bod_weights_df.index:
            bod_weights = bod_weights_df.loc[date]
        else:
            # If no weights are available for the current date, skip the risk calculations
            continue

        # Extract the covariance matrix for the current date
        cov_matrix = ewma_cov.loc[pd.IndexSlice[date, :], :].droplevel(0).values * 252

        # Calculate portfolio variance
        portfolio_variance = np.dot(bod_weights.values, np.dot(cov_matrix, bod_weights.values))
        df_risk.loc[date, 'Portfolio Variance'] = portfolio_variance

        # Calculate portfolio standard deviation
        portfolio_std_dev = np.sqrt(portfolio_variance)
        df_risk.loc[date, 'Portfolio Std Dev'] = portfolio_std_dev

        # Calculate Value at Risk (VaR) for the portfolio - assuming a normal distribution, 95% confidence level
        var_95 = -norm.ppf(0.05) * portfolio_std_dev
        df_risk.loc[date, 'VaR 95%'] = var_95

    for date in df_returns_subset.index:
        if date in bod_weights_df.index and date in ewma_cov.index.get_level_values(0):
            # Extract the begin-of-day weights for the current date
            bod_weights = bod_weights_df.loc[date]

            # Extract the covariance matrix for the current date
            cov_matrix = ewma_cov.xs(date, level='Date').reindex(index=bod_weights.index, columns=bod_weights.index) * 252

            # Ensure that the covariance matrix is square and matches the number of assets
            if not cov_matrix.shape[0] == cov_matrix.shape[1] == len(bod_weights):
                raise ValueError(f"Shape mismatch at date {date}: weights and covariance matrix must align.")

            # Calculate allocation risk
            allocation_risk = np.sqrt(np.dot(np.dot(bod_weights.values, cov_matrix.values), bod_weights.values))
            df_risk.loc[date, 'Allocation Risk'] = allocation_risk

            # Calculate marginal contributions to risk (MCR)
            mcr = np.dot(bod_weights.values, cov_matrix.values) / allocation_risk

            # Calculate contributions to risk (CTR)
            ctr = bod_weights.values * mcr

            # Calculate percentage contributions to risk (PCR)
            pcr = ctr / allocation_risk

            # Save the MCR, CTR, and PCR for each asset
            for i, asset in enumerate(bod_weights.index):
                df_risk.loc[date, f'MCR_{asset}'] = mcr[i]
                df_risk.loc[date, f'CTR_{asset}'] = ctr[i]
                df_risk.loc[date, f'PCR_{asset}'] = pcr[i]

    return df_risk


def main():
    # 1. Initialize session state
    if 'returns' not in st.session_state:
        st.session_state.returns = {}
    if 'portfolios' not in st.session_state:
        st.session_state.portfolios = {}
    if 'portfolios_risk' not in st.session_state:
        st.session_state.portfolios_risk = {}
    if 'benchmarks' not in st.session_state:
        st.session_state.benchmarks = {}
    if 'benchmarks_risk' not in st.session_state:
        st.session_state.benchmarks_risk = {}

    st.title("Portfolio Backtesting App")

    # 2. Uploading Data and Initial Configuration
    uploaded_file = st.sidebar.file_uploader("Choose an Excel file (.xlsx)", type="xlsx")
    if uploaded_file is not None:
        df_returns, df_weights = load_data(uploaded_file)
        choice = st.radio("Configure as:", ("Portfolio", "Benchmark"))

        # User Inputs for Configuration
        weight_option = st.selectbox("Select Weight Option:", ("Use Dataset Weights", "Custom Weights"))
        start_date = st.date_input("Select Start Date:", value=df_returns.index.min())
        name = st.text_input("Enter Name:")

        # Custom Weights Configuration
        if weight_option == "Custom Weights":
            custom_weights, rebalance_frequency = get_custom_weights_input(df_returns.columns)

        confirm_button = st.button("Confirm and Calculate")
        if confirm_button:
            strategic_weights = df_weights if weight_option == "Use Dataset Weights" else pd.DataFrame(custom_weights,
                                                                                                       index=[
                                                                                                           start_date])
            rebalanced_weights = generate_rebalanced_weights_for_frequency(strategic_weights, rebalance_frequency,
                                                                           start_date, df_returns)
            df_portfolio = calculate_detailed_portfolio(df_returns, rebalanced_weights, start_date, name)
            df_risk = calculate_risk_metrics(df_portfolio, df_returns, start_date, window=30)
            # 4. Store results in session state
            if choice == "Portfolio":
                st.session_state.returns = df_returns
                st.session_state.portfolios[name] = df_portfolio
                st.session_state.portfolios_risk[name] = df_risk

            else:
                st.session_state.benchmarks[name] = df_portfolio
                st.session_state.benchmarks_risk[name] = df_risk

    # 5. Analysis and Comparison
    if st.session_state.portfolios and st.session_state.benchmarks:
        portfolio_selection = st.selectbox("Select Portfolio for Analysis:", list(st.session_state.portfolios.keys()))
        benchmark_selection = st.selectbox("Select Benchmark for Comparison:", list(st.session_state.benchmarks.keys()))
        analyze_button = st.button("Analyze")
        reset_button = st.button("Reset")
        if analyze_button:
            # Assuming analysis functions are defined
            perform_analysis(st.session_state.portfolios[portfolio_selection],
                             st.session_state.benchmarks[benchmark_selection])
        elif reset_button:
            for key in st.session_state.keys():
                del st.session_state[key]
    else:
        st.write("Configure and confirm portfolios and benchmarks for analysis.")


def get_custom_weights_input(asset_columns):
    custom_weights = {}
    for asset in asset_columns:
        custom_weights[asset] = st.sidebar.slider(f"Weight for {asset} (%)", min_value=0.0, max_value=100.0, value=0.0,
                                                  step=1.0) / 100
        rebalance_frequency = st.sidebar.selectbox("Select Rebalancing Frequency:",
                                               ("Daily", "Weekly", "Monthly", "Quarterly", "Annually", "Buy and Hold"))
    return custom_weights, rebalance_frequency


def perform_analysis(portfolio_df, benchmark_df):
    for the_key in st.session_state.keys():
        st.write(the_key)

    for the_value in st.session_state.values():
        st.write(the_value)

    #st.write(st.session_state.portfolios, st.session_state.portfolios_risk, st.session_state.benchmarks, st.session_state.benchmarks_risk)

    if 'df_portfolio' in st.session_state['portfolios']:
        df_portfolio = st.session_state['df_portfolio']


    #create_plots(st.session_state.returns, st.session_state.portfolios, st.session_state.benchmarks,st.session_state.portfolios_risk)
    pass


def create_plots(df_returns, df_portfolio, df_benchmark, df_risk, start_date='31.12.2018'):
    # Plot 1: Cumulative Returns of Assets
    cum_returns = (1 + df_returns.loc[start_date:]).cumprod()
    fig1 = px.line(cum_returns, title='Cumulative Returns of Assets')
    st.plotly_chart(fig1)

    # Plot 2: Portfolio Value Over Time
    fig2 = px.line(df_portfolio, x=df_portfolio.index, y='Portfolio Index', title='Portfolio Value Over Time')
    # Check if df_benchmark is not None and add benchmark trace
    if df_benchmark is not None:
        # Convert Plotly Express figure to a Plotly Graph Objects Figure
        fig2 = go.Figure(fig1)
        # Add benchmark trace
        fig2.add_trace(
            go.Scatter(x=df_benchmark['Date'], y=df_benchmark.index, mode='lines', name='Benchmark'))

    st.plotly_chart(fig2)

    # Plot 3: Portfolio Standard Deviation Over Time
    fig3 = px.line(df_risk, x=df_risk.index, y='Portfolio Std Dev', title='Portfolio Standard Deviation')
    fig3.update_layout(xaxis_title='Date', yaxis_tickformat='.2%', yaxis_title='Standard Deviation')
    st.plotly_chart(fig3)

    # Plot 4: Daily portfolio returns as a bar chart
    fig4 = px.bar(df_portfolio, x=df_portfolio.index, y='Portfolio Return', title='Daily Portfolio Returns')
    fig4.update_layout(xaxis_title='Date', yaxis_tickformat='.2%', yaxis_title='Daily Return')
    st.plotly_chart(fig4)

    # Plot 5: Monthly Heatmap of Returns
    monthly_ret_table = ep.aggregate_returns(df_portfolio['Portfolio Return'].fillna(0.0), 'monthly')
    monthly_ret_table = monthly_ret_table.unstack().round(4) * 100
    fig5 = px.imshow(monthly_ret_table, text_auto=True, aspect="auto", color_continuous_scale="rdylgn")
    st.plotly_chart(fig5)

    # Plot 6: Weights as a stacked area chart
    # Identify weight columns that have non-zero values in any row
    non_zero_weight_cols = [col for col in df_portfolio.columns if
                            col.startswith('EOD_W_') and not (df_portfolio[col] == 0).all()]
    # Plot 5: Weights as a stacked area chart using filtered columns
    fig6 = px.area(df_portfolio, x=df_portfolio.index, y=non_zero_weight_cols, title='Asset Weights Over Time')
    fig6.update_layout(xaxis_title='Date', yaxis_tickformat='.2%', yaxis_title='Weights in Percent',
                       legend_title='Assets')
    st.plotly_chart(fig6)

    # Filter the columns to get only the PCS data
    # Filter columns where all values are np.nan or zero
    PCR_cols = [col for col in df_risk.columns if col.startswith('PCR_')]
    df_pcs = df_risk[PCR_cols].copy()
    df_pcs = df_pcs[1:]
    df_pcs.fillna(0, inplace=True)
    non_zero_pcs_cols = [col for col in df_pcs.columns if
                            col.startswith('PCR_') and not (df_pcs[col] == 0).all()]

    # Create an area chart using Plotly Express
    fig7 = px.area(df_pcs, x=df_pcs.index, y=non_zero_pcs_cols,
                  title='Daily Percentage Contribution to Risk (PCS) by Asset',
                  labels={'value': 'PCS', 'variable': 'Assets', 'index': 'Date'},
                  hover_data={'variable': False})
    # Optional: Customize the layout
    fig7.update_layout(xaxis_title='Date', yaxis_tickformat='.2%', yaxis_title='Percentage Contribution to Risk (PCR)', legend_title='Assets')
    st.plotly_chart(fig7)

    # Filter the columns to get only the CTR data
    CTR_cols = [col for col in df_risk.columns if col.startswith('CTR_')]
    df_ctr = df_risk[CTR_cols].copy()
    df_ctr.fillna(0, inplace=True)
    non_zero_ctr_cols = [col for col in df_ctr.columns if
                            col.startswith('CTR_') and not (df_ctr[col] == 0).all()]
    df_ctr = df_ctr[non_zero_ctr_cols]
    df_ctr = df_ctr[1:]
    # Create an area chart using Plotly Express
    fig8 = px.area(df_ctr, x=df_ctr.index, y=non_zero_ctr_cols, title='Daily Contribution to Risk (CTR) by Asset',
                   labels={'value': 'CTR', 'variable': 'Assets', 'index': 'Date'}, hover_data={'variable': False})
    # Optional: Customize the layout
    fig8.update_layout(xaxis_title='Date', yaxis_tickformat='.2%', yaxis_title='Contribution to Risk (CTR)', legend_title='Assets')
    st.plotly_chart(fig8)


if __name__ == '__main__':
    main()
