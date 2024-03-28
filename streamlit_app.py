import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
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

    if isinstance(strategic_weights, pd.Series):
        strategic_weights = strategic_weights.to_frame().T

    # Directly return the adjusted weights if using dataset weights
    if rebalance_frequency is None:
        return adjust_rebalancing_dates(strategic_weights, returns_df)

    closest_start_date = returns_df.index[min(range(len(returns_df.index)), key=lambda i: abs(returns_df.index[i] - start_date))]

    # Generate rebalance dates based on frequency
    rebalance_dates = pd.DatetimeIndex([])
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

    rebalanced_weights = pd.DataFrame(index=rebalance_dates, columns=strategic_weights.columns, dtype=float)
    for date in rebalance_dates:
        rebalanced_weights.loc[date] = strategic_weights.iloc[0].values

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

    if 'portfolio_name' not in st.session_state:
         st.session_state.portfolio_name = {}
    if 'benchmark_name' not in st.session_state:
         st.session_state.benchmark_name = {}

    st.title("Portfolio Backtesting App")

    # Initialize rebalance_frequency with a default value or as None
    rebalance_frequency = None

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
                st.session_state.portfolios_risk = df_risk
                st.session_state.portfolio_name = name

            else:
                st.session_state.benchmarks[name] = df_portfolio
                st.session_state.benchmarks_risk = df_risk
                st.session_state.benchmark_name = name

    # 5. Analysis and Comparison
    if st.session_state.portfolios and st.session_state.benchmarks:
        portfolio_selection = st.selectbox("Select Portfolio for Analysis:", list(st.session_state.portfolios.keys()))
        benchmark_selection = st.selectbox("Select Benchmark for Comparison:", list(st.session_state.benchmarks.keys()))
        analyze_button = st.button("Analyze")
        reset_button = st.button("Reset")
        if analyze_button:
            # Assuming analysis functions are defined
            perform_analysis(st.session_state.portfolios[portfolio_selection],
                             st.session_state.benchmarks[benchmark_selection],
                             st.session_state.returns,
                             st.session_state.portfolios_risk,
                             st.session_state.benchmarks_risk,
                             st.session_state.portfolio_name,
                             st.session_state.benchmark_name)
        elif reset_button:
            for key in st.session_state.keys():
                del st.session_state[key]
    else:
        st.write("Configure and confirm portfolios and benchmarks for analysis.")

def get_custom_weights_input(asset_columns):
    custom_weights = {}
    for i, asset in enumerate(asset_columns):
        # Use asset name as part of the key for slider to ensure uniqueness
        custom_weights[asset] = st.sidebar.slider(
            f"Weight for {asset} (%)", min_value=0.0, max_value=100.0, value=0.0, step=1.0, key=f"weight_{asset}"
        ) / 100

    # Use a unique key for the selectbox
    rebalance_frequency = st.selectbox(
        "Select Rebalancing Frequency:",
        ("Daily", "Weekly", "Monthly", "Quarterly", "Annually", "Buy and Hold"),
        key="rebalance_frequency"
    )
    return custom_weights, rebalance_frequency


def perform_analysis(portfolio_df, benchmark_df, returns_df, portfolio_risk, benchmark_risk, portfolio_name, benchmark_name):
    df_portfolio = pd.DataFrame(portfolio_df)
    df_benchmark = pd.DataFrame(benchmark_df)
    df_returns = pd.DataFrame(returns_df)
    df_portfolio_risk = pd.DataFrame(portfolio_risk)
    df_benchmark_risk = pd.DataFrame(benchmark_risk)

    create_plots(df_returns, df_portfolio, df_benchmark, df_portfolio_risk, df_benchmark_risk)

    pass


def create_plots(df_returns, df_portfolio, df_benchmark, df_portfolio_risk, df_benchmark_risk, start_date='31.12.2018'):
    # Plot 1: Cumulative Returns of Assets
    cum_returns = (1 + df_returns.loc[start_date:]).cumprod()
    fig1 = px.line(cum_returns, title='Cumulative Returns of Assets')
    st.plotly_chart(fig1)

    # Plot 2: Portfolio Value Over Time
    fig2 = go.Figure()
    # Add Portfolio trace
    fig2.add_trace(go.Scatter(x=df_portfolio.index, y=df_portfolio['Portfolio Index'],
                              mode='lines', name='Portfolio',
                              line=dict(color='blue')))
    # Check if df_benchmark is not None and add Benchmark trace
    if df_benchmark is not None:
        fig2.add_trace(go.Scatter(x=df_benchmark.index, y=df_benchmark['Portfolio Index'],
                                  mode='lines', name='Benchmark'))

    # Set titles and labels
    fig2.update_layout(title='Portfolio Value Over Time',
                       xaxis_title='Date',
                       yaxis_title='Index Value',
                       legend_title_text='Legend')
    st.plotly_chart(fig2)

    # Plot 3: Portfolio Standard Deviation Over Time
    fig3 = go.Figure()
    # Add Portfolio trace
    fig3.add_trace(go.Scatter(x=df_portfolio_risk.index, y=df_portfolio_risk['Portfolio Std Dev'],
                              mode='lines', name='Portfolio'))
    # Check if df_benchmark is not None and add Benchmark trace
    if df_benchmark is not None:
        fig3.add_trace(go.Scatter(x=df_benchmark_risk.index, y=df_benchmark_risk['Portfolio Std Dev'],
                                  mode='lines', name='Benchmark'))

    # Set titles and labels
    fig3.update_layout(title='Rolling Volatility Over Time',
                       xaxis_title='Date',
                       yaxis_title='Volatility in Percent',
                       yaxis_tickformat='.2%',
                       legend_title_text='Legend')
    # Show the plot in Streamlit
    st.plotly_chart(fig3)

    # Plot 4: Daily portfolio returns as a bar chart
    fig4 = go.Figure()
    # Add Portfolio trace
    fig4.add_trace(go.Bar(x=df_portfolio.index, y=df_portfolio['Portfolio Return'],
                          name='Portfolio'))

    # Check if df_benchmark is not None and add Benchmark bar trace
    if df_benchmark is not None:
        fig4.add_trace(go.Bar(x=df_benchmark.index, y=df_benchmark['Portfolio Return'],
                              name='Benchmark'))

    # Set titles and labels
    fig4.update_layout(title='Daily Returns',
                       xaxis_title='Date',
                       yaxis_title='Returns in Percent',
                       yaxis_tickformat='.2%',
                       barmode='group',  # Group bars for comparison if benchmark data is included
                       legend_title_text='Legend')

    # Show the plot in Streamlit
    st.plotly_chart(fig4)


    # Plot 4: Daily portfolio returns as a bar chart
    fig4a = go.Figure()
    # Add Portfolio trace
    fig4a.add_trace(go.Scatter(y=df_portfolio['Portfolio Return'], x=df_benchmark['Portfolio Return'],
                               showlegend=False,
                               mode='markers', marker=dict(
            color=df_portfolio['Portfolio Return'],  # set color equal to a variable
            colorscale="rdylgn",  # one of plotly colorscales,
            colorbar=dict(
                title='Portfolio Return %',
                tickformat='.2%'),  # Format tick labels as percentages
            showscale = True)))

    # Calculate the coefficients of the linear regression
    slope, intercept = np.polyfit(df_benchmark['Portfolio Return'], df_portfolio['Portfolio Return'], 1)
    # Generate the x-values for the regression line (from min to max of benchmark returns)
    x_reg_line = np.linspace(df_benchmark['Portfolio Return'].min(), df_benchmark['Portfolio Return'].max(), 100)
    # Calculate the y-values for the regression line
    y_reg_line = slope * x_reg_line + intercept

    # Display the regression equation as an annotation
    regression_equation = f'Trendline: Rᵖ= {intercept:.2f} + {slope:.2f}×Rᴮ'
    fig4a.add_annotation(xref='paper', yref='paper', x=0.05, y=0.95,
                         text=regression_equation,
                         showarrow=False, font=dict(size=14))

    # Add regression line trace
    fig4a.add_trace(go.Scatter(x=x_reg_line, y=y_reg_line, mode='lines', name='Regression Line',
                               showlegend=False,
                               line=dict(width=2)))

    # Set titles and labels
    fig4a.update_layout(title='Regression',
                       xaxis_title='Benchmark Returns',
                       yaxis_title='Portfolio Returns',
                       yaxis_tickformat='.2%', xaxis_tickformat='.2%',
                       legend_title_text='Legend')

    # Show the plot in Streamlit
    st.plotly_chart(fig4a)


    # Plot 5: Monthly Heatmap of Returns
    monthly_ret_table = ep.aggregate_returns(df_portfolio['Portfolio Return'].fillna(0.0), 'monthly')
    monthly_ret_table = monthly_ret_table.unstack().round(4) * 100
    fig5 = px.imshow(monthly_ret_table, text_auto=True, aspect="auto", color_continuous_scale="rdylgn",
                     title="Portfolio Monthly Heatmap")
    st.plotly_chart(fig5)

    # Plot 5: Monthly Heatmap of Returns
    if df_benchmark is not None:
        monthly_ret_table_bm = ep.aggregate_returns(df_benchmark['Portfolio Return'].fillna(0.0), 'monthly')
        monthly_ret_table_bm = monthly_ret_table_bm.unstack().round(4) * 100
        fig5a = px.imshow(monthly_ret_table_bm, text_auto=True, aspect="auto", color_continuous_scale="rdylgn",
                          title="Benchmark Monthly Heatmap")
        st.plotly_chart(fig5a)

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
    PCR_cols = [col for col in df_portfolio_risk.columns if col.startswith('PCR_')]
    df_pcs = df_portfolio_risk[PCR_cols].copy()
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
    CTR_cols = [col for col in df_portfolio_risk.columns if col.startswith('CTR_')]
    df_ctr = df_portfolio_risk[CTR_cols].copy()
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
