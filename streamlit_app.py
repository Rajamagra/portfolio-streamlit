#####################################################################################################
#Libraries: Check Requirements.txt
#####################################################################################################
import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import scipy.stats as stats
import empyrical as ep
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.regression.rolling import RollingOLS
from pypfopt import EfficientFrontier, expected_returns, risk_models


def load_data(uploaded_file):
    # Load the data from uploaded Excel file
    df_returns = pd.read_excel(uploaded_file, sheet_name='Returns', index_col=0, parse_dates=True)
    df_weights = pd.read_excel(uploaded_file, sheet_name='Weights', index_col=0, parse_dates=True)

    # PUBLICA Return
    xls = pd.ExcelFile(uploaded_file)
    if 'PUBLICA' in xls.sheet_names:
        df_publica = pd.read_excel(uploaded_file, sheet_name='PUBLICA', index_col=0, parse_dates=True)
        df_publica.columns = ['Portfolio Return']
        df_publica['Portfolio Index'] = ((1 + df_publica['Portfolio Return']).cumprod()) * 100
        return df_returns, df_weights, df_publica
        # The 'PUBLICA' sheet exists and df_publica is created
    else:
        return df_returns, df_weights, None


def generate_rebalanced_weights_for_frequency(strategic_weights, rebalance_frequency, start_date, returns_df):
    start_date = pd.to_datetime(start_date)

    if isinstance(strategic_weights, pd.Series):
        strategic_weights = strategic_weights.to_frame().T

    # Directly return the adjusted weights if using dataset weights
    if rebalance_frequency is None:
        return adjust_rebalancing_dates(strategic_weights, returns_df)

    closest_start_date = returns_df.index[
        min(range(len(returns_df.index)), key=lambda i: abs(returns_df.index[i] - start_date))]

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
    rebalancing_weights = adjust_rebalancing_dates(rebalancing_weights, df_returns)

    assets = df_returns.columns
    df_portfolio = pd.DataFrame(index=df_returns.index,
                                columns=['Rebalancing', 'Portfolio Return', 'Portfolio Index', 'Turnover'] +
                                        [f'BOD_W_{asset}' for asset in assets] +
                                        [f'EOD_W_{asset}' for asset in assets])
    df_portfolio.fillna(0, inplace=True)
    df_portfolio['Portfolio Index'] = 100.0

    eod_weights = rebalancing_weights.iloc[0].values if not rebalancing_weights.empty else np.zeros(len(assets))

    for i, date in enumerate(df_returns.index):
        is_rebalancing_day = date in rebalancing_weights.index
        df_portfolio.loc[date, 'Rebalancing'] = is_rebalancing_day

        if i > 0:
            previous_date = df_returns.index[i - 1]
            df_portfolio.loc[date, [f'BOD_W_{asset}' for asset in assets]] = df_portfolio.loc[
                previous_date, [f'EOD_W_{asset}' for asset in assets]].values

        bod_weights = df_portfolio.loc[date, [f'BOD_W_{asset}' for asset in assets]].values.astype(float)
        portfolio_return = np.dot(bod_weights, df_returns.loc[date].values)
        df_portfolio.loc[date, 'Portfolio Return'] = portfolio_return

        if i > 0:
            prev_index = df_portfolio.loc[previous_date, 'Portfolio Index']
            df_portfolio.loc[date, 'Portfolio Index'] = prev_index * (1 + portfolio_return)

        if not is_rebalancing_day:
            daily_returns = df_returns.loc[date].values
            eod_weights *= (1 + daily_returns)
            eod_weights /= eod_weights.sum()

        df_portfolio.loc[date, [f'EOD_W_{asset}' for asset in assets]] = eod_weights

        # Calculate turnover if there is a rebalancing event
        if is_rebalancing_day and i > 0:
            turnover = np.sum(np.abs(
                eod_weights - df_portfolio.loc[previous_date, [f'EOD_W_{asset}' for asset in assets]].values)) / 2
            df_portfolio.loc[date, 'Turnover'] = turnover
        else:
            df_portfolio.loc[date, 'Turnover'] = 0

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
        var_95 = -stats.norm.ppf(0.05) * portfolio_std_dev
        df_risk.loc[date, 'VaR 95%'] = var_95

    for date in df_returns_subset.index:
        if date in bod_weights_df.index and date in ewma_cov.index.get_level_values(0):
            # Extract the begin-of-day weights for the current date
            bod_weights = bod_weights_df.loc[date]

            # Extract the covariance matrix for the current date
            cov_matrix = ewma_cov.xs(date, level='Date').reindex(index=bod_weights.index,
                                                                 columns=bod_weights.index) * 252

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


def display_custom_weights_figure(custom_weights):
    total_weight = sum(custom_weights.values())
    weights_df = pd.DataFrame({
        'Asset': list(custom_weights.keys()),
        'Weight': list(custom_weights.values())
    })
    fig = px.bar(weights_df, x='Asset', y='Weight', title='Custom Weights Distribution')
    fig.add_hline(y=total_weight, line_dash="dot",
                  annotation_text=f"Total Weight: {total_weight * 100:.2f}%",
                  annotation_position="bottom right")
    st.plotly_chart(fig)


def main():
    st.set_page_config(layout="wide")

    # 1. Initialize session state
    if 'analyzed' not in st.session_state:
        st.session_state.analyzed = False  # Flag to track if analysis has been performed
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

    st.header("Portfolio Backtesting Tool", divider='gray')

    # Initialize variables outside of the sidebar scope
    rebalance_frequency = None
    choice = None
    name = None
    confirm_button = False
    custom_weights = None  # Initialize custom_weights

    # Sidebar for file uploading and initial configuration
    with st.sidebar:
        uploaded_file = st.file_uploader("Choose an Excel file (.xlsx)", type="xlsx")
        if uploaded_file is not None:
            df_returns, df_weights, df_publica = load_data(uploaded_file)

            with st.expander("Configuration", expanded=True):
                choice = st.radio("Configure as:", ("Portfolio", "Benchmark", "PUBLICA") if df_publica is not None else ("Portfolio", "Benchmark"))
                weight_option = st.selectbox("Select Weight Option:", ("Use Dataset Weights", "Custom Weights"))
                start_date = st.date_input("Select Start Date:", value=df_returns.index.min())

                if choice == "Portfolio":
                    name = st.text_input("Enter Name, default is Portfolio:", "Portfolio")
                elif choice == "Benchmark":
                    name = st.text_input("Enter name, default is Benchmark:", "Benchmark")
                elif choice == "PUBLICA":
                    name = st.text_input("Enter Name default is PUBLICA", "PUBLICA")

                if weight_option == "Custom Weights":
                    custom_weights, rebalance_frequency = get_custom_weights_input(df_returns.columns)

                confirm_button = st.button("Confirm and Calculate")

    if confirm_button and choice is not None:
        if choice == "Portfolio":
            with st.spinner('Calculating Portfolio, wait for it...'):
                strategic_weights = df_weights if weight_option == "Use Dataset Weights" else pd.DataFrame(custom_weights, index=[start_date])
                rebalanced_weights = generate_rebalanced_weights_for_frequency(strategic_weights, rebalance_frequency, start_date, df_returns)
                df_portfolio = calculate_detailed_portfolio(df_returns, rebalanced_weights, start_date, name)
                df_risk = calculate_risk_metrics(df_portfolio, df_returns, start_date, window=30)

    # Visualize custom weights distribution on the main page
    if custom_weights is not None:
        total_weight = sum(custom_weights.values())
        weights_df = pd.DataFrame({'Asset': custom_weights.keys(), 'Weight': custom_weights.values()})

        if not st.session_state.analyzed:
            fig = px.bar(weights_df, x='Asset', y='Weight', title='Custom Weights Distribution')
            fig.add_hline(y=total_weight, line_dash="dot", annotation_text=f"Total Weight: {total_weight * 100:.2f}%", annotation_position="bottom right")
            st.plotly_chart(fig)
            if total_weight > 1:
                st.error(f"Total Weight: {total_weight * 100:.2f}%. The total weight exceeds 100%.")
            elif total_weight < 1:
                st.warning(f"Total Weight: {total_weight * 100:.2f}%. The total weight is less than 100%.")

    if confirm_button and choice is not None:
        if choice == "Benchmark":
            with st.spinner('Calculating Benchmark, wait for it...'):
                strategic_weights = df_weights if weight_option == "Use Dataset Weights" else pd.DataFrame(custom_weights,
                                                                                                           index=[start_date])
                rebalanced_weights = generate_rebalanced_weights_for_frequency(strategic_weights, rebalance_frequency,
                                                                           start_date, df_returns)
                df_portfolio = calculate_detailed_portfolio(df_returns, rebalanced_weights, start_date, name)
                df_risk = calculate_risk_metrics(df_portfolio, df_returns, start_date, window=30)

        st.success('Done!')

        # Store results in session state
        if choice == "Portfolio":
            st.session_state.returns = df_returns
            st.session_state.portfolios[name] = df_portfolio
            st.session_state.portfolios_risk = df_risk
            st.session_state.portfolio_name = name
        elif choice == "PUBLICA":
            st.session_state.benchmarks[name] = df_publica

            # Dummy df_risk for PUBLICA
            df_risk = pd.DataFrame({
                'Portfolio Std Dev': df_publica['Portfolio Return'].rolling(window=30).std() * np.sqrt(252),
                'VaR 95%': -stats.norm.ppf(0.05) * df_publica['Portfolio Return'].rolling(window=30).std() * np.sqrt(252)},
                index=df_publica.index)

            st.session_state.benchmarks_risk = df_risk
            st.session_state.benchmark_name = "PUBLICA"

        else:  # Benchmark
            st.session_state.benchmarks[name] = df_portfolio
            st.session_state.benchmarks_risk = df_risk
            st.session_state.benchmark_name = name

    # Analysis and Comparison
    if st.session_state.portfolios and st.session_state.benchmarks:
        with st.expander("Choose a Portfolio and a Benchmark:", expanded=True):
            portfolio_selection = st.selectbox("Select Portfolio for Analysis:", list(st.session_state.portfolios.keys()))
            benchmark_selection = st.selectbox("Select Benchmark for Comparison:", list(st.session_state.benchmarks.keys()))
            analyze_button = st.button("Analyze")
            reset_button = st.button("Reset")

            st.session_state.analyzed = True  # Set the flag to True when analyzed

        if analyze_button:
                perform_analysis(st.session_state.portfolios[portfolio_selection],
                             st.session_state.benchmarks[benchmark_selection],
                             st.session_state.returns,
                             st.session_state.portfolios_risk,
                             st.session_state.benchmarks_risk,
                             st.session_state.portfolio_name,
                             st.session_state.benchmark_name)
        elif reset_button:
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.session_state.analyzed = False  # Reset the analyzed flag

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


def perform_analysis(portfolio_df, benchmark_df, returns_df, portfolio_risk, benchmark_risk, portfolio_name,
                     benchmark_name):
    df_portfolio = pd.DataFrame(portfolio_df)
    df_benchmark = pd.DataFrame(benchmark_df)
    df_returns = pd.DataFrame(returns_df)
    df_portfolio_risk = pd.DataFrame(portfolio_risk)
    df_benchmark_risk = pd.DataFrame(benchmark_risk)
    create_plots(df_returns, df_portfolio, df_benchmark, df_portfolio_risk, df_benchmark_risk, portfolio_name, benchmark_name)
    pass


def create_plots(df_returns, df_portfolio, df_benchmark, df_portfolio_risk, df_benchmark_risk, portfolio_name,
                 benchmark_name, start_date='31.12.2018'):

    def value_at_risk(returns, period=None, sigma=2.0):
        """
        Get value at risk (VaR).

        Parameters
        ----------
        returns : pd.Series
            Daily returns of the strategy, noncumulative.
             - See full explanation in tears.create_full_tear_sheet.
        period : str, optional
            Period over which to calculate VaR. Set to 'weekly',
            'monthly', or 'yearly', otherwise defaults to period of
            returns (typically daily).
        sigma : float, optional
            Standard deviations of VaR, default 2.
        """
        if period is not None:
            returns_agg = ep.aggregate_returns(returns, period)
        else:
            returns_agg = returns.copy()

        value_at_risk = returns_agg.mean() - sigma * returns_agg.std()
        return value_at_risk

    def perf_stats(returns, factor_returns=None):
        # Portfolio Statistics:
        SIMPLE_STAT_FUNCS = [
            ep.annual_return,
            ep.cum_returns_final,
            ep.annual_volatility,
            ep.sharpe_ratio,
            ep.calmar_ratio,
            ep.stability_of_timeseries,
            ep.max_drawdown,
            ep.omega_ratio,
            ep.sortino_ratio,
            stats.skew,
            stats.kurtosis,
            ep.tail_ratio,
            value_at_risk]
        FACTOR_STAT_FUNCS = [
            ep.alpha,
            ep.beta, ]
        STAT_FUNC_NAMES = {
            'annual_return': 'Annual return',
            'cum_returns_final': 'Cumulative returns',
            'annual_volatility': 'Annual volatility',
            'sharpe_ratio': 'Sharpe ratio',
            'calmar_ratio': 'Calmar ratio',
            'stability_of_timeseries': 'Stability',
            'max_drawdown': 'Max drawdown',
            'omega_ratio': 'Omega ratio',
            'sortino_ratio': 'Sortino ratio',
            'skew': 'Skew',
            'kurtosis': 'Kurtosis',
            'tail_ratio': 'Tail ratio',
            'common_sense_ratio': 'Common sense ratio',
            'value_at_risk': 'Daily value at risk',
            'alpha': 'Alpha',
            'beta': 'Beta', }
        STAT_FUNCS_PCT = [
            'Annual return',
            'Cumulative returns',
            'Annual volatility',
            'Max drawdown',
            'Daily value at risk',
            'Daily turnover'
        ]

        statistics = pd.Series()
        for stat_func in SIMPLE_STAT_FUNCS:
            statistics[STAT_FUNC_NAMES[stat_func.__name__]] = stat_func(returns)

        if factor_returns is not None:
            for stat_func in FACTOR_STAT_FUNCS:
                res = stat_func(returns, factor_returns)
                statistics[STAT_FUNC_NAMES[stat_func.__name__]] = res
        return statistics

    stats_portfolio = perf_stats(df_portfolio['Portfolio Return'], factor_returns=df_benchmark['Portfolio Return'])
    stats_benchmark = perf_stats(df_benchmark['Portfolio Return'], factor_returns = None)
    df_stats = pd.concat([stats_portfolio, stats_benchmark], axis=1)
    df_stats.columns = [str(portfolio_name), str(benchmark_name)]

    # PORTFOLIO STATS
    with st.container(border=True, ):
        st.subheader("Stats", divider = "gray")
        st.dataframe(df_stats, use_container_width=True)

    # FIG 1: Cumulative Returns of Assets
    cum_returns = (1 + df_returns.loc[start_date:]).cumprod()
    fig1 = px.line(cum_returns, title='Cumulative Returns of Assets')
    st.plotly_chart(fig1, use_container_width=True)

    # FIG 2: Portfolio Value Over Time
    fig2 = go.Figure()
    # Add Portfolio trace
    fig2.add_trace(go.Scatter(x=df_portfolio.index, y=df_portfolio['Portfolio Index'],
                              mode='lines', name=portfolio_name,
                              line=dict(color='blue')))
    # Check if df_benchmark is not None and add Benchmark trace
    if df_benchmark is not None:
        fig2.add_trace(go.Scatter(x=df_benchmark.index, y=df_benchmark['Portfolio Index'],
                                  mode='lines', name=benchmark_name))

    # Set titles and labels
    fig2.update_layout(title=f'{portfolio_name} and {benchmark_name} Value Over Time',
                       xaxis_title='Date',
                       yaxis_title='Index Value',
                       legend_title_text='Legend')
    st.plotly_chart(fig2, use_container_width=True)

    # TWO COLUMNS FOR OUTPUT
    c1, c2 = st.columns((1, 1))

    # FIG 3: Monthly Portfolio and Benchmark Return as a bar chart
    df_monthly = pd.DataFrame({
        'Portfolio': df_portfolio['Portfolio Index'],
        'Benchmark': df_benchmark['Portfolio Index']}, index=df_portfolio.index)
    df_monthly = df_monthly.resample('M').last()
    df_monthly = df_monthly.pct_change().dropna()
    fig3 = go.Figure()
    # Add Portfolio trace
    fig3.add_trace(go.Bar(x=df_monthly.index, y=df_monthly['Portfolio'], name=f'{portfolio_name}'))
    fig3.add_trace(go.Bar(x=df_monthly.index, y=df_monthly['Benchmark'], name=f'{benchmark_name}'))
    # Set titles and labels
    fig3.update_layout(title=f'{portfolio_name} and {benchmark_name} Monthly Returns',
                       xaxis_title='Date',
                       yaxis_title='Returns in Percent',
                       yaxis_tickformat='.2%',
                       barmode='group',  # Group bars for comparison if benchmark data is included
                       legend_title_text='Legend')
    c1.plotly_chart(fig3, use_container_width=True)

    # FIG 4: Daily Portfolio and Benchmark Return as a bar chart
    fig4 = go.Figure()
    # Add Portfolio trace
    fig4.add_trace(go.Bar(x=df_portfolio.index, y=df_portfolio['Portfolio Return'],
                          name=f'{portfolio_name}'))
    # Check if df_benchmark is not None and add Benchmark bar trace
    if df_benchmark is not None:
        fig4.add_trace(go.Bar(x=df_benchmark.index, y=df_benchmark['Portfolio Return'],
                              name=f'{benchmark_name}'))
    # Set titles and labels
    fig4.update_layout(title=f'{portfolio_name} and {benchmark_name} Daily Returns',
                       xaxis_title='Date',
                       yaxis_title='Returns in Percent',
                       yaxis_tickformat='.2%',
                       barmode='group',  # Group bars for comparison if benchmark data is included
                       legend_title_text='Legend')
    c2.plotly_chart(fig4, use_container_width=True)

    # FIG 5: Portfolio Monthly Heatmap of Returns
    monthly_ret_table = ep.aggregate_returns(df_portfolio['Portfolio Return'].fillna(0.0), 'monthly')
    monthly_ret_table = monthly_ret_table.unstack().round(4) * 100
    fig5 = px.imshow(monthly_ret_table, text_auto=True, aspect="auto", color_continuous_scale="rdylgn",
                     title=f'{portfolio_name} Monthly Returns Heatmap')
    c1.plotly_chart(fig5, use_container_width=True)

    # FIG 6: Benchmark Monthly Heatmap of Returns
    if df_benchmark is not None:
        monthly_ret_table_bm = ep.aggregate_returns(df_benchmark['Portfolio Return'].fillna(0.0), 'monthly')
        monthly_ret_table_bm = monthly_ret_table_bm.unstack().round(4) * 100
        fig6 = px.imshow(monthly_ret_table_bm, text_auto=True, aspect="auto", color_continuous_scale="rdylgn",
                         title=f'{benchmark_name} Monthly Returns Heatmap')
    c2.plotly_chart(fig6, use_container_width=True)

    # FIG 7: Standard Deviation Over Time
    fig7 = go.Figure()
    # Add Portfolio trace
    fig7.add_trace(go.Scatter(x=df_portfolio_risk.index, y=df_portfolio_risk['Portfolio Std Dev'],
                              mode='lines', name=f'{portfolio_name}'))
    # Check if df_benchmark is not None and add Benchmark trace
    if df_benchmark is not None:
        fig7.add_trace(go.Scatter(x=df_benchmark_risk.index, y=df_benchmark_risk['Portfolio Std Dev'],
                                  mode='lines', name=f'{benchmark_name}'))

    # Set titles and labels
    fig7.update_layout(title='Rolling Volatility Over Time',
                       xaxis_title='Date',
                       yaxis_title='Volatility in Percent',
                       yaxis_tickformat='.2%',
                       legend_title_text='Legend')
    c1.plotly_chart(fig7, use_container_width=True)

    # FIG 8: Tracking Error Portfolio vs. Benchmark
    # Step 1: Calculate differences
    return_diff = df_portfolio['Portfolio Return'] - df_benchmark['Portfolio Return']
    # Step 2: Calculate rolling Tracking Error
    rolling_te = return_diff.rolling(window=30).std() * np.sqrt(252)
    # Step 3: Calculate overall Tracking Error
    overall_te = return_diff.std() * np.sqrt(252)
    # Step 4: Plotting
    fig8 = go.Figure()
    fig8.add_trace(go.Scatter(x=rolling_te.index, y=rolling_te, mode='lines', name='Rolling Tracking Error'))
    fig8.add_hline(y=overall_te, line_dash="dot", annotation_text=f"Overall Tracking Error: {overall_te * 100:.2f}%",
                   annotation_position="bottom right", line_width=5, line_color='purple')

    fig8.update_layout(title="Rolling Tracking Error with Overall Tracking Error",
                       xaxis_title="Date",
                       yaxis_title="Tracking Error",
                       yaxis_tickformat='.2%',
                       legend_title="Legend")
    c2.plotly_chart(fig8, use_container_width=True)

    # FIG 9: 95% Value at Risk
    fig9 = go.Figure()
    # Add Portfolio trace
    fig9.add_trace(go.Scatter(y=df_portfolio_risk['VaR 95%'], x=df_portfolio_risk.index,
                              mode='lines', name=f'{portfolio_name}'))
    fig9.add_trace(go.Scatter(x=df_portfolio_risk.index, y=df_benchmark_risk['VaR 95%'],
                              mode='lines', name=f'{benchmark_name}'))
    fig9.update_layout(
        title='Value at Risk 95%',
        xaxis_title="Date",
        yaxis_title=f'Value at Risk in % of {portfolio_name}',
        legend_title="Legend",
        yaxis_tickformat='.2%')
    c1.plotly_chart(fig9, use_container_width=True)

    # FIG10: Histogram of Returns
    fig10 = go.Figure()
    fig10.add_trace(go.Histogram(x=df_portfolio['Portfolio Return'], name=f'{portfolio_name}'))
    fig10.add_trace(go.Histogram(x=df_benchmark['Portfolio Return'], name=f'{benchmark_name}'))
    fig10.update_layout(
        title='Histogram of Returns',
        xaxis_title="Daily Return",
        yaxis_title="Frequency",
        legend_title="Legend",
        xaxis_tickformat='.2%')

    # Calculate Value at Risk (VaR) for the portfolio - assuming a normal distribution, 95% confidence level
    var_95_Port = -stats.norm.ppf(0.05) * df_portfolio['Portfolio Return'].std()
    fig10.add_trace(go.Scatter(x=[var_95_Port, var_95_Port], y=[0, 250],
                               mode='lines', name='Value at Risk 95%',
                               line=dict(width=1, color='red', dash='dash')))

    # Overlay both histograms
    fig10.update_layout(barmode='overlay')
    # Reduce opacity to see both histograms
    fig10.update_traces(opacity=0.75)
    # Show the plot in Streamlit
    c2.plotly_chart(fig10, use_container_width=True)

    # FIG 11: Daily Returns Portfolio vs. Benchmark + Regression
    fig11 = go.Figure()
    # Add Portfolio trace
    fig11.add_trace(go.Scatter(y=df_portfolio['Portfolio Return'], x=df_benchmark['Portfolio Return'],
                              showlegend=False, mode='markers', marker=dict(
            color=df_portfolio['Portfolio Return'],  # set color equal to a variable
            colorscale="rdylgn",  # one of plotly colorscales,
            colorbar=dict(
                title='Portfolio Return %',
                tickformat='.2%'),  # Format tick labels as percentages
            showscale=True)))

    # Calculate the coefficients of the linear regression
    slope, intercept = np.polyfit(df_benchmark['Portfolio Return'], df_portfolio['Portfolio Return'], 1)
    # Generate the x-values for the regression line (from min to max of benchmark returns)
    x_reg_line = np.linspace(df_benchmark['Portfolio Return'].min(), df_benchmark['Portfolio Return'].max(), 100)
    # Calculate the y-values for the regression line
    y_reg_line = slope * x_reg_line + intercept

    # Display the regression equation as an annotation
    regression_equation = f'Trendline: Rᵖ= {intercept:.2f} + {slope:.2f}×Rᴮ'
    fig11.add_annotation(xref='paper', yref='paper', x=0.05, y=0.95,
                        text=regression_equation, showarrow=False, font=dict(size=14))

    # Add regression line trace
    fig11.add_trace(go.Scatter(x=x_reg_line, y=y_reg_line, mode='lines', name='Regression Line',
                              showlegend=False, line=dict(width=2)))

    # Set titles and labels
    fig11.update_layout(title='Regression',
                       xaxis_title=f'{benchmark_name} Returns',
                       yaxis_title=f'{portfolio_name} Returns',
                       yaxis_tickformat='.2%', xaxis_tickformat='.2%',
                       legend_title_text='Legend')
    c1.plotly_chart(fig11, use_container_width=True)

    # FIG 12 - Rolling Beta 6M/12M
    def calculate_and_plot_rolling_beta(portfolio_returns, benchmark_returns):
        # Ensure the index is in datetime format
        portfolio_returns.index = pd.to_datetime(portfolio_returns.index)
        benchmark_returns.index = pd.to_datetime(benchmark_returns.index)

        # Merge the two dataframes on their indices (dates)
        combined_df = pd.concat([portfolio_returns, benchmark_returns], axis=1)
        combined_df.columns = ['portfolio_returns', 'benchmark_returns']

        # Add a constant to the independent variable for OLS regression
        combined_df['const'] = 1

        # Calculate overall beta for the entire horizon
        overall_model = sm.OLS(combined_df['portfolio_returns'], combined_df[['benchmark_returns', 'const']]).fit()
        overall_beta = overall_model.params['benchmark_returns']

        # Calculate rolling beta for 6 months (126 trading days) and 12 months (252 trading days)
        rolling_beta_6m = RollingOLS(combined_df['portfolio_returns'], combined_df[['benchmark_returns', 'const']],
                                     window=126).fit().params['benchmark_returns']
        rolling_beta_12m = RollingOLS(combined_df['portfolio_returns'], combined_df[['benchmark_returns', 'const']],
                                      window=252).fit().params['benchmark_returns']

        # Create a Plotly figure
        fig = go.Figure()

        # Add 6 months rolling beta line
        fig.add_trace(go.Scatter(x=rolling_beta_6m.index, y=rolling_beta_6m, mode='lines', name='6-Month Rolling Beta'))

        # Add 12 months rolling beta line
        fig.add_trace(
            go.Scatter(x=rolling_beta_12m.index, y=rolling_beta_12m, mode='lines', name='12-Month Rolling Beta'))

        # Add overall beta horizontal line
        fig.add_trace(go.Scatter(x=[combined_df.index.min(), combined_df.index.max()], y=[overall_beta, overall_beta],
                                 mode='lines', name='Overall Beta', line=dict(dash='dot')))

        # Update layout
        fig.update_layout(title='Rolling Beta of Portfolio against Benchmark',
                          xaxis_title='Date',
                          yaxis_title='Beta',
                          legend_title='Beta Type')
        return fig

    fig12 = calculate_and_plot_rolling_beta(df_portfolio['Portfolio Return'], df_benchmark['Portfolio Return'])
    c2.plotly_chart(fig12, use_container_width=True)

    # FIG 13 - Max Drawdown Portfolio
    def gen_drawdown_table(returns, top=3):
        """
        Places top drawdowns in a table.

        Parameters
        ----------
        returns : pd.Series
            Daily returns of the strategy, noncumulative.
             - See full explanation in tears.create_full_tear_sheet.
        top : int, optional
            The amount of top drawdowns to find (default 10).

        Returns
        -------
        df_drawdowns : pd.DataFrame
            Information about top drawdowns.
        """

        df_cum = ep.cum_returns(returns, 1.0)
        drawdown_periods = get_top_drawdowns(returns, top=top)
        df_drawdowns = pd.DataFrame(index=list(range(top)),
                                    columns=['Net drawdown in %',
                                             'Peak date',
                                             'Valley date',
                                             'Recovery date',
                                             'Duration'])

        for i, (peak, valley, recovery) in enumerate(drawdown_periods):
            if pd.isnull(recovery):
                df_drawdowns.loc[i, 'Duration'] = np.nan
            else:
                df_drawdowns.loc[i, 'Duration'] = len(pd.date_range(peak,
                                                                    recovery,
                                                                    freq='B'))
            df_drawdowns.loc[i, 'Peak date'] = (peak.to_pydatetime()
                                                .strftime('%Y-%m-%d'))
            df_drawdowns.loc[i, 'Valley date'] = (valley.to_pydatetime()
                                                  .strftime('%Y-%m-%d'))
            if isinstance(recovery, float):
                df_drawdowns.loc[i, 'Recovery date'] = recovery
            else:
                df_drawdowns.loc[i, 'Recovery date'] = (recovery.to_pydatetime()
                                                        .strftime('%Y-%m-%d'))
            df_drawdowns.loc[i, 'Net drawdown in %'] = (
                                                               (df_cum.loc[peak] - df_cum.loc[valley]) / df_cum.loc[
                                                           peak]) * 100

        df_drawdowns['Peak date'] = pd.to_datetime(df_drawdowns['Peak date'])
        df_drawdowns['Valley date'] = pd.to_datetime(df_drawdowns['Valley date'])
        df_drawdowns['Recovery date'] = pd.to_datetime(df_drawdowns['Recovery date'])

        return df_drawdowns

    def get_max_drawdown_underwater(underwater):
        """
        Determines peak, valley, and recovery dates given an 'underwater'
        DataFrame.

        An underwater DataFrame is a DataFrame that has precomputed
        rolling drawdown.

        Parameters
        ----------
        underwater : pd.Series
           Underwater returns (rolling drawdown) of a strategy.

        Returns
        -------
        peak : datetime
            The maximum drawdown's peak.
        valley : datetime
            The maximum drawdown's valley.
        recovery : datetime
            The maximum drawdown's recovery.
        """

        valley = underwater.idxmin()  # end of the period
        # Find first 0
        peak = underwater[:valley][underwater[:valley] == 0].index[-1]
        # Find last 0
        try:
            recovery = underwater[valley:][underwater[valley:] == 0].index[0]
        except IndexError:
            recovery = np.nan  # drawdown not recovered
        return peak, valley, recovery

    def get_top_drawdowns(returns, top=3):
        """
        Finds top drawdowns, sorted by drawdown amount.

        Parameters
        ----------
        returns : pd.Series
            Daily returns of the strategy, noncumulative.
             - See full explanation in tears.create_full_tear_sheet.
        top : int, optional
            The amount of top drawdowns to find (default 10).

        Returns
        -------
        drawdowns : list
            List of drawdown peaks, valleys, and recoveries. See get_max_drawdown.
        """
        returns = returns.copy()
        df_cum = ep.cum_returns(returns, 1.0)
        running_max = np.maximum.accumulate(df_cum)
        underwater = df_cum / running_max - 1

        drawdowns = []
        for _ in range(top):
            peak, valley, recovery = get_max_drawdown_underwater(underwater)
            # Slice out draw-down period
            if not pd.isnull(recovery):
                underwater.drop(underwater[peak: recovery].index[1:-1],
                                inplace=True)
            else:
                # drawdown has not ended yet
                underwater = underwater.loc[:peak]

            drawdowns.append((peak, valley, recovery))
            if ((len(returns) == 0)
                    or (len(underwater) == 0)
                    or (np.min(underwater) == 0)):
                break
        return drawdowns

    def plot_drawdown_periods_from_portfolio(df_portfolio, portfolio_name, top=3):
        # Assuming 'Portfolio Return' is the correct column name
        returns = df_portfolio['Portfolio Return']

        # Generate cumulative returns and drawdown table
        df_cum_rets = ep.cum_returns(returns, starting_value=1.0)
        df_drawdowns = gen_drawdown_table(returns, top=top)

        # Create the base line plot for cumulative returns
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(x=df_cum_rets.index, y=df_cum_rets, name='Cumulative Returns', line=dict(color='blue')))

        # Add shaded areas for top drawdown periods
        for i, row in df_drawdowns.iterrows():
            if pd.isnull(row['Recovery date']):
                recovery_date = returns.index[-1]  # If not recovered, use the last date
            else:
                recovery_date = row['Recovery date']
            fig.add_vrect(x0=row['Peak date'], x1=recovery_date,
                          annotation_text=f"Drawdown {i + 1}", annotation_position="top left",
                          fillcolor="red", opacity=0.3, line_width=0)

        # Customize the layout
        fig.update_layout(title=f'{portfolio_name}: Top {top} Drawdown Periods',
                          xaxis_title='Date',
                          yaxis_title='Cumulative Returns',
                          legend_title='Legend')
        return fig

    fig13 = plot_drawdown_periods_from_portfolio(df_portfolio, portfolio_name, top=3)
    c1.plotly_chart(fig13, use_container_width=True)

    # FIG 14 - Drawdown Benchmark
    fig14 = plot_drawdown_periods_from_portfolio(df_benchmark, benchmark_name, top=3)
    c2.plotly_chart(fig14, use_container_width=True)

    # FIG 15 - Percentage of Risk
    # Filter the columns to get only the PCS data
    # Filter columns where all values are np.nan or zero
    PCR_cols = [col for col in df_portfolio_risk.columns if col.startswith('PCR_')]
    df_pcs = df_portfolio_risk[PCR_cols].copy()
    # Assuming `df` has a DateTimeIndex
    df_pcs = df_pcs[df_pcs.index.dayofweek < 5]
    df_pcs = df_pcs[5:]
    df_pcs.fillna(0, inplace=True)
    non_zero_pcs_cols = [col for col in df_pcs.columns if
                         col.startswith('PCR_') and not (df_pcs[col] == 0).all()]

    # Create an area chart using Plotly Express
    fig15 = px.bar(df_pcs, x=df_pcs.index, y=non_zero_pcs_cols,
                  title='Daily Percentage Contribution to Risk (PCS) by Asset',
                  labels={'value': 'PCS', 'variable': 'Assets', 'index': 'Date'},
                  hover_data={'variable': False})
    # Optional: Customize the layout
    fig15.update_layout(xaxis_title='Date', yaxis_tickformat='.2%', yaxis_title='Percentage Contribution to Risk (PCR)',
                       legend_title='Assets')
    c2.plotly_chart(fig15, use_container_width=True)

    # FIG 16 - Contribution to Total Risk
    # Filter the columns to get only the CTR data
    CTR_cols = [col for col in df_portfolio_risk.columns if col.startswith('CTR_')]
    df_ctr = df_portfolio_risk[CTR_cols].copy()
    df_ctr.fillna(0, inplace=True)
    non_zero_ctr_cols = [col for col in df_ctr.columns if
                         col.startswith('CTR_') and not (df_ctr[col] == 0).all()]
    df_ctr = df_ctr[non_zero_ctr_cols]
    df_ctr = df_ctr[1:]
    # Create an area chart using Plotly Express
    fig16 = px.area(df_ctr, x=df_ctr.index, y=non_zero_ctr_cols, title='Daily Contribution to Risk (CTR) by Asset',
                   labels={'value': 'CTR', 'variable': 'Assets', 'index': 'Date'}, hover_data={'variable': False})
    # Optional: Customize the layout
    fig16.update_layout(xaxis_title='Date', yaxis_tickformat='.2%', yaxis_title='Contribution to Risk (CTR)',
                       legend_title='Assets')
    c1.plotly_chart(fig16, use_container_width=True)

    # FIG 17 - Correlation Heatmap for Portfolio
    from plotly.subplots import make_subplots

    def create_rolling_correlation_heatmap(df_returns, window):
        # Calculate rolling correlation matrices
        rolling_corr = df_returns.rolling(window=window).corr(pairwise=True)

        # Extract unique dates for animation frames
        dates = rolling_corr.index.get_level_values(0).unique()

        # Initialize figure with subplots
        fig = make_subplots(rows=1, cols=1, subplot_titles=('Rolling Correlation Heatmap',))

        # Create frames for animation
        frames = []

        for date in dates:
            # Extract correlation matrix for the current date
            corr_matrix = rolling_corr.loc[date].values

            # Skip the date if correlation matrix is all NaN (insufficient data)
            if np.isnan(corr_matrix).all():
                continue

            # Make the matrix asymmetric by omitting the first asset from columns
            corr_matrix_asymmetric = corr_matrix[:, 1:]
            asset_labels = df_returns.columns[1:]

            # Create heatmap for the current date
            heatmap = go.Heatmap(
                z=corr_matrix_asymmetric,
                x=asset_labels,
                y=df_returns.columns,  # Keep all assets in rows
                colorscale='RdYlGn_r',  # Inverted Red to Green color scale
                zmin=-1,  # Minimum correlation value
                zmax=1,  # Maximum correlation value
                text=np.around(corr_matrix_asymmetric, 2),  # Round correlations to two digits and convert to text
                texttemplate="%{text}",
                hoverinfo="text"  # Show text on hover
            )

            frame_name = date.strftime('%Y-%m-%d')  # Format date
            frames.append(go.Frame(data=[heatmap], name=frame_name,
                                   layout=go.Layout(title_text='Rolling Correlation Heatmap: ' + frame_name)))

        # Set the initial state with the first frame's data
        fig.add_trace(frames[0].data[0])
        fig.update_layout(title_text='Rolling Correlation Heatmap: ' + frames[0].name)

        # Add Play/Pause button and slider to the layout
        fig.update_layout(
            updatemenus=[{
                "type": "buttons",
                "showactive": False,
                "buttons": [{
                    "label": "Play",
                    "method": "animate",
                    "args": [None,
                             {"frame": {"duration": 500, "redraw": True}, "fromcurrent": True, "mode": "immediate"}]
                }, {
                    "label": "Pause",
                    "method": "animate",
                    "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}]
                }],
                "direction": "left",
                "pad": {"r": 10, "t": 87},
                "x": 1.0,
                "xanchor": "right",
                "y": 1.15,  # Position slightly above the top right corner
                "yanchor": "top"
            }],
            sliders=[{
                "steps": [{
                    "method": 'animate',
                    "args": [[frame.name], {"frame": {"duration": 500, "redraw": True}, "mode": "immediate"}],
                    "label": frame.name
                } for frame in frames],
                "currentvalue": {"prefix": "Date: "},
                "x": 0.1,
                "len": 0.9,
                "xanchor": "left",
                "y": -.1,  # Keep the slider positioning
            }],
            xaxis_title='Assets',
            yaxis_title='Assets',
            coloraxis_colorbar=dict(title='Correlation'),
        )

        # Add frames to the figure
        fig.frames = frames

        return fig


    # FIG 17 - Correlation Heatmap for Portfolio
    from plotly.subplots import make_subplots

    def create_rolling_correlation_heatmap(df_returns, window):
        # Calculate rolling correlation matrices
        rolling_corr = df_returns.rolling(window=window).corr(pairwise=True)

        # Extract unique dates for animation frames
        dates = rolling_corr.index.get_level_values(0).unique()

        # Initialize figure with subplots
        fig = make_subplots(rows=1, cols=1, subplot_titles=('Rolling Correlation Heatmap',))

        # Create frames for animation
        frames = []

        for date in dates:
            # Extract correlation matrix for the current date
            corr_matrix = rolling_corr.loc[date].values

            # Skip the date if correlation matrix is all NaN (insufficient data)
            if np.isnan(corr_matrix).all():
                continue

            # Make the matrix asymmetric by omitting the first asset from columns
            corr_matrix_asymmetric = corr_matrix[:, 1:]
            asset_labels = df_returns.columns[1:]

            # Create heatmap for the current date
            heatmap = go.Heatmap(
                z=corr_matrix_asymmetric,
                x=asset_labels,
                y=df_returns.columns,  # Keep all assets in rows
                colorscale='RdYlGn_r',  # Inverted Red to Green color scale
                zmin=-1,  # Minimum correlation value
                zmax=1,  # Maximum correlation value
                text=np.around(corr_matrix_asymmetric, 2),  # Round correlations to two digits and convert to text
                texttemplate="%{text}",
                hoverinfo="text"  # Show text on hover
            )

            frame_name = date.strftime('%Y-%m-%d')  # Format date
            frames.append(go.Frame(data=[heatmap], name=frame_name,
                                   layout=go.Layout(title_text='Rolling Correlation Heatmap: ' + frame_name)))

        # Set the initial state with the first frame's data
        fig.add_trace(frames[0].data[0])
        fig.update_layout(title_text='Rolling Correlation Heatmap: ' + frames[0].name)

        # Add Play/Pause button and slider to the layout
        fig.update_layout(
            updatemenus=[{
                "type": "buttons",
                "showactive": False,
                "buttons": [{
                    "label": "Play",
                    "method": "animate",
                    "args": [None,
                             {"frame": {"duration": 500, "redraw": True}, "fromcurrent": True, "mode": "immediate"}]
                }, {
                    "label": "Pause",
                    "method": "animate",
                    "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}]
                }],
                "direction": "left",
                "pad": {"r": 10, "t": 87},
                "x": 1.0,
                "xanchor": "right",
                "y": 1.15,  # Position slightly above the top right corner
                "yanchor": "top"
            }],
            sliders=[{
                "steps": [{
                    "method": 'animate',
                    "args": [[frame.name], {"frame": {"duration": 500, "redraw": True}, "mode": "immediate"}],
                    "label": frame.name
                } for frame in frames],
                "currentvalue": {"prefix": "Date: "},
                "x": 0.1,
                "len": 0.9,
                "xanchor": "left",
                "y": -.1,  # Keep the slider positioning
            }],
            xaxis_title='Assets',
            yaxis_title='Assets',
            coloraxis_colorbar=dict(title='Correlation'),
        )

        # Add frames to the figure
        fig.frames = frames

        return fig

    fig17 = create_rolling_correlation_heatmap(df_returns, window=30)
    st.plotly_chart(fig17, use_container_width=True)

    # FIG 18: Weights as a stacked area chart
    # Identify weight columns that have non-zero values in any row
    non_zero_weight_cols = [col for col in df_portfolio.columns if
                            col.startswith('EOD_W_') and not (df_portfolio[col] == 0).all()]
    # Plot 5: Weights as a stacked area chart using filtered columns
    fig18 = px.area(df_portfolio, x=df_portfolio.index, y=non_zero_weight_cols, title='Asset Weights Over Time')
    fig18.update_layout(xaxis_title='Date', yaxis_tickformat='.2%', yaxis_title='Weights in Percent',
                       legend_title='Assets')
    st.plotly_chart(fig18, use_container_width=True)

    # FIG 19: Efficient Frontier
    # Convert daily returns to cumulative returns
    cumulative_returns = (1 + df_returns).cumprod()
    # Resample to get end-of-month values and then calculate monthly returns from these
    monthly_cumulative_returns = cumulative_returns.resample('M').last()
    monthly_returns = monthly_cumulative_returns.pct_change().dropna()

    # Calculate expected returns and sample covariance
    mu = expected_returns.mean_historical_return(monthly_returns, returns_data=True,
                                                 compounding=True, frequency=12, log_returns=False)
    S = risk_models.CovarianceShrinkage(monthly_returns, returns_data=True, frequency=12,
                                        log_returns=False).ledoit_wolf().astype(float)

    # Minimum Variance Portfolio
    ef = EfficientFrontier(mu, S)  # Re-initialize to reset the weights
    weights_min_var = ef.min_volatility()
    perf_min_var = ef.portfolio_performance(verbose=True)
    del ef
    # Maximum Sharpe Ratio (Tangency Portfolio)
    ef = EfficientFrontier(mu, S)  # Re-initialize to reset the weights
    weights_max_sharpe = ef.max_sharpe()
    perf_max_sharpe = ef.portfolio_performance(verbose=True)

    # Maximum Return Portfolio: Identifying the asset with the maximum return
    max_return_asset = mu.idxmax()
    # Calculate the standard deviation (volatility) of the maximum return asset
    volatility_max_return = np.sqrt(S.loc[max_return_asset, max_return_asset])
    # Set the weights to be 100% in the max return asset
    weights_max_return = {asset: 0.0 for asset in mu.index}
    weights_max_return[max_return_asset] = 1.0
    # Performance metrics for the maximum return portfolio
    perf_max_return = (
        mu[max_return_asset], volatility_max_return, 'N/A')  # Return, Volatility, Sharpe (N/A without risk-free rate)

    # Calculate efficient frontier again
    returns_range = np.linspace(perf_min_var[0] + 0.0001, perf_max_return[0] - 0.0001, 20)
    volatility_range = []

    for r in returns_range:
        ef = EfficientFrontier(mu, S)
        ef.add_constraint(lambda w: w >= 0)
        ef.add_constraint(lambda w: sum(w) == 1)
        ef.efficient_return(target_return=r, market_neutral=False)
        volatility_range.append(ef.portfolio_performance()[1])

    fig = go.Figure()
    # Efficient Frontier
    fig.add_trace(go.Scatter(x=volatility_range, y=returns_range, mode='lines', name='Efficient Frontier'))

    # Add Special Portfolios
    fig.add_trace(go.Scatter(x=[(perf_min_var[1])], y=[perf_min_var[0]],
                             mode='markers', marker_symbol='x', name='Min Variance'))
    fig.add_trace(go.Scatter(x=[(perf_max_sharpe[1])], y=[perf_max_sharpe[0]],
                             mode='markers', marker_symbol='x', name='Tangency'))
    fig.add_trace(go.Scatter(x=[perf_max_return[1]], y=[perf_max_return[0]],
                             mode='markers', marker_symbol='x', name='Max Return'))

    # Assets
    for asset in mu.index:
        fig.add_trace(go.Scatter(x=[np.sqrt(S.loc[asset, asset])], y=[mu[asset]],
                                 mode='markers', name=asset))

    fig.update_layout(xaxis_title="Volatility (Std. Deviation)",
                      xaxis_tickformat='.2%',
                      yaxis_title="Expected Return",
                      yaxis_tickformat='.2%',
                      title="Efficient Frontier with Special Portfolios")
    fig.update_traces(marker_size=12)
    st.plotly_chart(fig, use_container_width=True)

    # Portfolio Optimization over Horizon
    with st.spinner('Calculating Efficient Frontiers...'):
        st.write("Rolling Portfolio Optimization")
        # Convert daily returns to cumulative returns
        cumulative_returns = (1 + df_returns).cumprod()
        monthly_cumulative_returns = cumulative_returns.resample('M').last()
        monthly_cumulative_returns = monthly_cumulative_returns.pct_change().dropna()
        df_MVP_W, df_MVP, df_MSP_W, df_EF = rolling_portfolio_optimization(monthly_cumulative_returns, 24)
    st.success('Done!')

    # Efficient Frontier
    df_EF_reset = df_EF.reset_index(inplace=False)
    fig8 = px.scatter(df_EF_reset, x=df_EF_reset.Volatility, y=df_EF_reset.Return, animation_frame=df_EF_reset.Date,
                      color=df_EF_reset.Date,
                      hover_name=df_EF_reset.Date, range_x=[0, df_EF_reset.Volatility.max()],
                      range_y=[0, df_EF_reset.Return.max()])
    fig8.update_layout(xaxis_title='Volatility', xaxis_tickformat='.2%', yaxis_tickformat='.2%', yaxis_title='Return',
                       template='seaborn')

    st.plotly_chart(fig8, use_container_width=True)

    fig9 = px.bar(df_MVP_W, x=df_MVP_W.index, y=df_MVP_W.columns)
    fig9.update_layout(xaxis_title="Weights of Assets (Minimum Volatility Portfolio)",
                       yaxis_title="Allocation in Percent",
                       yaxis_tickformat='.2%',
                       title="Minimum Volatility Portfolio Weights")
    st.plotly_chart(fig9, use_container_width=True)

    # FIG 20: Maximum Sharpe Portfolio Allocation
    fig20 = px.bar(df_MSP_W, x=df_MSP_W.index, y=df_MSP_W.columns)
    fig20.update_layout(xaxis_title="Strategy Weights (Maximum Sharpe Portfolio)",
                       yaxis_title="Allocation in Percent",
                       yaxis_tickformat='.2%',
                       title="Maximum Sharpe Portfolio Weights")
    st.plotly_chart(fig20, use_container_width=True)


        


if __name__ == '__main__':
    main()
