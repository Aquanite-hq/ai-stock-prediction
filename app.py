import streamlit as st

# Set page config - Must be the first Streamlit command
st.set_page_config(
    page_title="AI Stock Market Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import yfinance as yf
import time

# Import local modules
from data_loader import load_stock_data, get_company_info, get_market_indices
from technical_indicators import calculate_technical_indicators
from prediction_models import predict_stock_prices
from portfolio import Portfolio
from visualizations import (
    plot_stock_chart, 
    plot_prediction_chart, 
    plot_indicators_chart,
    plot_comparison_chart,
    plot_portfolio_performance
)
from utils import format_number, calculate_percentage_change, get_market_status
from database import init_app
from auth import initialize_auth, show_login_page, is_logged_in, get_current_user_id, logout_user

# Initialize app and get default user
try:
    with st.spinner("Connecting to database..."):
        default_user = init_app()
except Exception as e:
    st.error(f"Error initializing application: {e}")
    default_user = type('obj', (object,), {'id': 1, 'username': 'system'})

# Initialize authentication
initialize_auth()

# Add custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        font-weight: bold;
        color: #1E88E5;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #0D47A1;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    .info-text {
        font-size: 1rem;
        color: #546E7A;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .buy-signal {
        color: green;
        font-weight: bold;
    }
    .sell-signal {
        color: red;
        font-weight: bold;
    }
    /* Make the metric values black */
    [data-testid="stMetricValue"] {
        color: black !important;
    }
</style>
""", unsafe_allow_html=True)

# Session state initialization
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = Portfolio()

if 'comparison_tickers' not in st.session_state:
    st.session_state.comparison_tickers = []

# Check if user is logged in
if not is_logged_in():
    # Show login page
    show_login_page()
else:
    # User is logged in, show the main application
    # Create sidebar
    st.sidebar.markdown("<div class='main-header'>AI Stock Predictor</div>", unsafe_allow_html=True)
    
    # Add logout button
    if st.sidebar.button("Logout"):
        logout_user()
        st.rerun()
    
    # Show username
    st.sidebar.markdown(f"<div class='info-text'>Logged in as: <b>{st.session_state.username}</b></div>", unsafe_allow_html=True)
    
    # Sidebar sections
    sidebar_section = st.sidebar.radio(
        "Navigation",
        [
            "📊 Dashboard",
            "🔍 Stock Analysis",
            "🤖 AI Predictions",
            "📈 Technical Indicators", 
            "💼 Portfolio Tracker",
            "🔎 Stock Comparison",
        ],
        key="navigation"
    )

    # Main content
    if sidebar_section == "📊 Dashboard":
        # -------- Dashboard Section --------
        st.markdown("<div class='main-header'>Market Dashboard</div>", unsafe_allow_html=True)
        
        # Market status
        market_status = get_market_status()
        st.markdown(f"<div class='info-text'>Market Status: <b>{market_status}</b></div>", unsafe_allow_html=True)
        
        # Display market indices in columns
        indices_data = get_market_indices()
        
        if not indices_data.empty:
            cols = st.columns(len(indices_data))
            
            for i, (_, row) in enumerate(indices_data.iterrows()):
                with cols[i]:
                    change_color = "green" if row['Change'] >= 0 else "red"
                    st.markdown(f"""
                    <div class='metric-card'>
                        <div style='font-size: 1.2rem; font-weight: bold; color: black;'>{row['Index']}</div>
                        <div style='font-size: 1.5rem; color: black;'>{format_number(row['Price'], 2)}</div>
                        <div style='color: {change_color};'>
                            {format_number(row['Change'], 2)} ({format_number(row['Change %'], 2)}%)
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        
        # Quick search for a stock
        st.markdown("<div class='sub-header'>Quick Stock Search</div>", unsafe_allow_html=True)
        quick_search = st.text_input("Enter a stock ticker:", value="AAPL")
        
        if quick_search:
            quick_search = quick_search.upper().strip()
            
            # Load stock data
            stock_data = load_stock_data(quick_search, period="1mo")
            
            if not stock_data.empty:
                # Display stock chart
                st.plotly_chart(plot_stock_chart(stock_data, quick_search), use_container_width=True)
                
                # Display key metrics
                st.markdown("<div class='sub-header'>Key Metrics</div>", unsafe_allow_html=True)
                
                # Get company info
                company_info = get_company_info(quick_search)
                
                # Create columns for metrics
                metric_cols = st.columns(4)
                
                with metric_cols[0]:
                    st.metric(
                        label="Current Price", 
                        value=f"${format_number(stock_data['Close'].iloc[-1], 2)}",
                        delta=f"{format_number(stock_data['Close'].iloc[-1] - stock_data['Close'].iloc[-2], 2)}",
                        label_visibility="visible"
                    )
                
                with metric_cols[1]:
                    st.metric(
                        label="Market Cap",
                        value=f"${format_number(company_info.get('market_cap', 0) / 1e9, 2)}B"
                    )
                
                with metric_cols[2]:
                    st.metric(
                        label="P/E Ratio",
                        value=format_number(company_info.get('pe_ratio', 0), 2)
                    )
                
                with metric_cols[3]:
                    st.metric(
                        label="Dividend Yield",
                        value=f"{format_number(company_info.get('dividend_yield', 0) * 100, 2)}%"
                    )

    elif sidebar_section == "🔍 Stock Analysis":
        # -------- Stock Analysis Section --------
        st.markdown("<div class='main-header'>Stock Analysis</div>", unsafe_allow_html=True)
        
        # Input for stock ticker and period
        ticker = st.text_input("Enter Stock Ticker Symbol:", value="AAPL").upper()
        
        col1, col2 = st.columns(2)
        with col1:
            period = st.selectbox(
                "Select Period:",
                ["1mo", "3mo", "6mo", "1y", "2y", "5y", "max"],
                index=3
            )
        
        with col2:
            interval = st.selectbox(
                "Select Interval:",
                ["1d", "5d", "1wk", "1mo"],
                index=0
            )
        
        if ticker:
            # Load stock data
            stock_data = load_stock_data(ticker, period=period)
            
            if not stock_data.empty:
                # Display stock chart
                st.plotly_chart(plot_stock_chart(stock_data, ticker), use_container_width=True)
                
                # Display company information
                st.markdown("<div class='sub-header'>Company Information</div>", unsafe_allow_html=True)
                
                company_info = get_company_info(ticker)
                
                if company_info:
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        st.markdown("<div class='info-text'>Key Stats</div>", unsafe_allow_html=True)
                        stats_data = {
                            "Market Cap": f"${format_number(company_info.get('market_cap', 0) / 1e9, 2)}B",
                            "P/E Ratio": format_number(company_info.get('pe_ratio', 0), 2),
                            "Dividend Yield": f"{format_number(company_info.get('dividend_yield', 0) * 100, 2)}%",
                            "Sector": company_info.get('sector', 'N/A'),
                            "Industry": company_info.get('industry', 'N/A')
                        }
                        
                        for key, value in stats_data.items():
                            st.markdown(f"**{key}:** {value}")
                    
                    with col2:
                        st.markdown("<div class='info-text'>Description</div>", unsafe_allow_html=True)
                        st.markdown(company_info.get('description', 'No description available.'))
                
                # Display financial data
                st.markdown("<div class='sub-header'>Price History</div>", unsafe_allow_html=True)
                
                # Create columns for metrics
                metric_cols = st.columns(5)
                
                with metric_cols[0]:
                    current_price = stock_data['Close'].iloc[-1]
                    previous_close = stock_data['Close'].iloc[-2]
                    price_change = current_price - previous_close
                    price_change_pct = (price_change / previous_close) * 100
                    
                    st.metric(
                        label="Current Price",
                        value=f"${format_number(current_price, 2)}",
                        delta=f"{format_number(price_change, 2)} ({format_number(price_change_pct, 2)}%)"
                    )
                
                with metric_cols[1]:
                    st.metric(
                        label="Open",
                        value=f"${format_number(stock_data['Open'].iloc[-1], 2)}"
                    )
                
                with metric_cols[2]:
                    st.metric(
                        label="High",
                        value=f"${format_number(stock_data['High'].iloc[-1], 2)}"
                    )
                
                with metric_cols[3]:
                    st.metric(
                        label="Low",
                        value=f"${format_number(stock_data['Low'].iloc[-1], 2)}"
                    )
                
                with metric_cols[4]:
                    st.metric(
                        label="Volume",
                        value=f"{format_number(stock_data['Volume'].iloc[-1], 0)}"
                    )
                
                # Display historical performance
                st.markdown("<div class='sub-header'>Historical Performance</div>", unsafe_allow_html=True)
                
                performance_cols = st.columns(4)
                
                # Calculate returns for different periods
                current_price = stock_data['Close'].iloc[-1]
                
                # 1 week return
                one_week_ago = stock_data['Close'].iloc[-6] if len(stock_data) >= 6 else stock_data['Close'].iloc[0]
                one_week_return = ((current_price - one_week_ago) / one_week_ago) * 100
                
                # 1 month return
                one_month_ago = stock_data['Close'].iloc[-22] if len(stock_data) >= 22 else stock_data['Close'].iloc[0]
                one_month_return = ((current_price - one_month_ago) / one_month_ago) * 100
                
                # 3 month return
                three_months_ago = stock_data['Close'].iloc[-66] if len(stock_data) >= 66 else stock_data['Close'].iloc[0]
                three_month_return = ((current_price - three_months_ago) / three_months_ago) * 100
                
                # YTD return
                start_of_year = stock_data.loc[stock_data.index.year == datetime.now().year].iloc[0]['Close'] if any(idx.year == datetime.now().year for idx in stock_data.index) else stock_data['Close'].iloc[0]
                ytd_return = ((current_price - start_of_year) / start_of_year) * 100
                
                with performance_cols[0]:
                    st.metric(
                        label="1 Week",
                        value=f"{format_number(one_week_return, 2)}%",
                        delta=f"{format_number(one_week_return, 2)}%"
                    )
                
                with performance_cols[1]:
                    st.metric(
                        label="1 Month",
                        value=f"{format_number(one_month_return, 2)}%",
                        delta=f"{format_number(one_month_return, 2)}%"
                    )
                
                with performance_cols[2]:
                    st.metric(
                        label="3 Months",
                        value=f"{format_number(three_month_return, 2)}%",
                        delta=f"{format_number(three_month_return, 2)}%"
                    )
                
                with performance_cols[3]:
                    st.metric(
                        label="YTD",
                        value=f"{format_number(ytd_return, 2)}%",
                        delta=f"{format_number(ytd_return, 2)}%"
                    )

    elif sidebar_section == "🤖 AI Predictions":
        # -------- AI Predictions Section --------
        st.markdown("<div class='main-header'>AI Stock Price Predictions</div>", unsafe_allow_html=True)
        
        st.markdown("""
        <div class='info-text'>
        This section uses machine learning algorithms to predict future stock prices.
        Choose a stock ticker and prediction settings below.
        </div>
        """, unsafe_allow_html=True)
        
        # Input for stock ticker and prediction settings
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            ticker = st.text_input("Enter Stock Ticker Symbol:", value="MSFT").upper()
        
        with col2:
            prediction_days = st.slider("Prediction Days:", min_value=5, max_value=90, value=30, step=5)
        
        with col3:
            model_type = st.selectbox(
                "Prediction Model:",
                ["Prophet", "LSTM", "ARIMA", "Ensemble"],
                index=0
            )
        
        # Load historical data
        if ticker:
            # Get longer historical data for better prediction
            historical_data = load_stock_data(ticker, period="2y")
            
            if not historical_data.empty:
                # Display historical chart
                st.markdown("<div class='sub-header'>Historical Price Chart</div>", unsafe_allow_html=True)
                st.plotly_chart(plot_stock_chart(historical_data, ticker), use_container_width=True)
                
                # Run prediction
                with st.spinner(f"Running {model_type} prediction for {ticker}..."):
                    try:
                        predicted_data, metrics = predict_stock_prices(
                            historical_data, 
                            model_type=model_type,
                            prediction_days=prediction_days
                        )
                        
                        if predicted_data.empty:
                            st.error(f"Failed to generate prediction for {ticker}. Please try a different model or stock.")
                            predicted_data = pd.DataFrame({
                                'ds': pd.Series(pd.date_range(start=historical_data.index[-1] + timedelta(days=1), periods=prediction_days)),
                                'yhat': pd.Series([historical_data['Close'].iloc[-1]] * prediction_days)
                            })
                            metrics = {'mae': 0, 'mse': 0, 'rmse': 0, 'confidence': 50}
                    
                    except Exception as e:
                        st.error(f"Error in prediction: {e}")
                        # Create fallback prediction data
                        predicted_data = pd.DataFrame({
                            'ds': pd.Series(pd.date_range(start=historical_data.index[-1] + timedelta(days=1), periods=prediction_days)),
                            'yhat': pd.Series([historical_data['Close'].iloc[-1]] * prediction_days)
                        })
                        metrics = {'mae': 0, 'mse': 0, 'rmse': 0, 'confidence': 50}
                
                if 'yhat' in predicted_data.columns:
                    # Display prediction chart
                    st.markdown("<div class='sub-header'>Price Prediction</div>", unsafe_allow_html=True)
                    st.plotly_chart(plot_prediction_chart(historical_data, predicted_data, ticker), use_container_width=True)
                    
                    # Display prediction metrics
                    st.markdown("<div class='sub-header'>Prediction Metrics</div>", unsafe_allow_html=True)
                    
                    metric_cols = st.columns(4)
                    
                    with metric_cols[0]:
                        st.metric(
                            label="Predicted Price (End)",
                            value=f"${format_number(predicted_data['yhat'].iloc[-1], 2)}"
                        )
                    
                    with metric_cols[1]:
                        start_price = historical_data['Close'].iloc[-1]
                        end_price = predicted_data['yhat'].iloc[-1]
                        price_change = end_price - start_price
                        price_change_pct = (price_change / start_price) * 100
                        
                        st.metric(
                            label="Price Change",
                            value=f"${format_number(price_change, 2)}",
                            delta=f"{format_number(price_change_pct, 2)}%"
                        )
                    
                    with metric_cols[2]:
                        st.metric(
                            label="Confidence Score",
                            value=f"{format_number(metrics['confidence'], 2)}%"
                        )
                    
                    with metric_cols[3]:
                        st.metric(
                            label="RMSE",
                            value=format_number(metrics['rmse'], 2)
                        )
                    
                    # Display predicted values table
                    st.markdown("<div class='sub-header'>Predicted Prices by Date</div>", unsafe_allow_html=True)
                    
                    # Filter to only show future predictions
                    future_predictions = predicted_data[predicted_data['ds'] > historical_data.index[-1]]
                    
                    # Create a more readable dataframe for display
                    display_df = pd.DataFrame({
                        'Date': future_predictions['ds'],
                        'Predicted Price': future_predictions['yhat']
                    })
                    
                    # Add percentage change column
                    display_df['Change %'] = display_df['Predicted Price'].pct_change() * 100
                    if len(display_df) > 0:
                        display_df['Change %'].iloc[0] = ((display_df['Predicted Price'].iloc[0] / historical_data['Close'].iloc[-1]) - 1) * 100
                    
                    st.dataframe(display_df.set_index('Date'), use_container_width=True)
                else:
                    st.error("Prediction model couldn't generate valid forecast data. Please try a different model or stock.")
                    st.markdown("### Prediction Not Available")
                    st.write("The selected model is unable to generate predictions for this stock. Please try one of the following:")
                    st.write("- Select a different prediction model")
                    st.write("- Choose a different stock")
                    st.write("- Use a different time period")
                
                # Model explanation
                st.markdown("<div class='sub-header'>Model Explanation</div>", unsafe_allow_html=True)
                
                model_explanations = {
                    "Prophet": "Facebook's Prophet is a time series forecasting model that works well with data that has strong seasonal patterns and can handle missing data points.",
                    "LSTM": "Long Short-Term Memory (LSTM) is a type of recurrent neural network that can learn and remember long-term dependencies in time series data.",
                    "ARIMA": "Autoregressive Integrated Moving Average (ARIMA) is a statistical model that uses past data to predict future values and is effective for non-seasonal time series.",
                    "Ensemble": "Ensemble combines predictions from multiple models to create a more robust forecast, often resulting in better accuracy than any individual model."
                }
                
                st.info(model_explanations.get(model_type, "Model explanation not available."))
                
                # Add prediction disclaimer
                st.warning("""
                **Disclaimer**: Stock price predictions are based on historical data and mathematical models.
                They should not be used as the sole basis for investment decisions. Always conduct thorough research and consider consulting with a financial advisor.
                Past performance is not indicative of future results.
                """)

    elif sidebar_section == "📈 Technical Indicators":
        # -------- Technical Indicators Section --------
        st.markdown("<div class='main-header'>Technical Indicators</div>", unsafe_allow_html=True)
        
        # Input for stock ticker and period
        ticker = st.text_input("Enter Stock Ticker Symbol:", value="AAPL").upper()
        
        period = st.selectbox(
            "Select Period:",
            ["1mo", "3mo", "6mo", "1y", "2y", "5y"],
            index=2
        )
        
        if ticker:
            # Load stock data
            stock_data = load_stock_data(ticker, period=period)
            
            if not stock_data.empty:
                # Calculate technical indicators
                indicators = calculate_technical_indicators(stock_data)
                
                # Display technical indicators chart
                st.plotly_chart(plot_indicators_chart(stock_data, indicators), use_container_width=True)
                
                # Display trading signals
                st.markdown("<div class='sub-header'>Trading Signals</div>", unsafe_allow_html=True)
                
                signals = indicators.get('signals', {})
                current_price = stock_data['Close'].iloc[-1]
                
                # Create columns for different signal types
                signal_cols = st.columns(3)
                
                with signal_cols[0]:
                    st.markdown("#### Moving Averages")
                    
                    ma_signals = signals.get('moving_averages', {})
                    
                    for ma_name, signal in ma_signals.items():
                        signal_class = "buy-signal" if signal == "Buy" else "sell-signal" if signal == "Sell" else "neutral-signal"
                        st.markdown(f"{ma_name}: <span class='{signal_class}'>{signal}</span>", unsafe_allow_html=True)
                
                with signal_cols[1]:
                    st.markdown("#### Oscillators")
                    
                    oscillator_signals = signals.get('oscillators', {})
                    
                    for osc_name, signal in oscillator_signals.items():
                        signal_class = "buy-signal" if signal == "Buy" else "sell-signal" if signal == "Sell" else "neutral-signal"
                        st.markdown(f"{osc_name}: <span class='{signal_class}'>{signal}</span>", unsafe_allow_html=True)
                
                with signal_cols[2]:
                    st.markdown("#### Trend Indicators")
                    
                    trend_signals = signals.get('trend', {})
                    
                    for trend_name, signal in trend_signals.items():
                        signal_class = "buy-signal" if signal == "Buy" else "sell-signal" if signal == "Sell" else "neutral-signal"
                        st.markdown(f"{trend_name}: <span class='{signal_class}'>{signal}</span>", unsafe_allow_html=True)
                
                # Display overall signal summary
                # Count bullish, bearish, and neutral signals
                buy_count = sum(1 for signal, value in signals.items() if ('above' in signal or 'golden' in signal or 'oversold' in signal or 'crossover' in signal) and value)
                sell_count = sum(1 for signal, value in signals.items() if ('below' in signal or 'death' in signal or 'overbought' in signal or 'crossunder' in signal) and value)
                neutral_count = len(signals) - buy_count - sell_count
                
                total_signals = buy_count + sell_count + neutral_count
                buy_percentage = (buy_count / total_signals) * 100 if total_signals > 0 else 0
                sell_percentage = (sell_count / total_signals) * 100 if total_signals > 0 else 0
                
                st.markdown("<div class='sub-header'>Signal Summary</div>", unsafe_allow_html=True)
                summary_cols = st.columns(3)
                
                with summary_cols[0]:
                    st.metric("Buy Signals", f"{buy_count} ({format_number(buy_percentage, 0)}%)")
                
                with summary_cols[1]:
                    st.metric("Sell Signals", f"{sell_count} ({format_number(sell_percentage, 0)}%)")
                
                with summary_cols[2]:
                    st.metric("Neutral Signals", f"{neutral_count} ({format_number(100 - buy_percentage - sell_percentage, 0)}%)")
                
                # Display indicator values
                st.markdown("<div class='sub-header'>Indicator Values</div>", unsafe_allow_html=True)
                
                # Create tabs for different indicator categories
                indicator_tabs = st.tabs(["Moving Averages", "Oscillators", "Volatility", "Support/Resistance"])
                
                with indicator_tabs[0]:
                    # Moving Averages
                    st.markdown("#### Moving Average Values")
                    
                    ma_data = {
                        "SMA 20": format_number(indicators['sma20'].iloc[-1], 2),
                        "SMA 50": format_number(indicators['sma50'].iloc[-1], 2),
                        "SMA 200": format_number(indicators['sma200'].iloc[-1], 2),
                        "EMA 12": format_number(indicators['ema12'].iloc[-1], 2),
                        "EMA 26": format_number(indicators['ema26'].iloc[-1], 2)
                    }
                    
                    ma_table = pd.DataFrame([ma_data])
                    st.table(ma_table)
                
                with indicator_tabs[1]:
                    # Oscillators
                    st.markdown("#### Oscillator Values")
                    
                    oscillator_data = {
                        "RSI (14)": format_number(indicators['rsi'].iloc[-1], 2),
                        "MACD": format_number(indicators['macd'].iloc[-1], 2),
                        "MACD Signal": format_number(indicators['macd_signal'].iloc[-1], 2),
                        "MACD Histogram": format_number(indicators['macd_histogram'].iloc[-1], 2),
                        "Stochastic %K": format_number(indicators['stoch_k'].iloc[-1], 2),
                        "Stochastic %D": format_number(indicators['stoch_d'].iloc[-1], 2)
                    }
                    
                    oscillator_table = pd.DataFrame([oscillator_data])
                    st.table(oscillator_table)
                
                with indicator_tabs[2]:
                    # Volatility
                    st.markdown("#### Volatility Indicators")
                    
                    volatility_data = {
                        "Bollinger Upper": format_number(indicators['bollinger_upper'].iloc[-1], 2),
                        "Bollinger Middle": format_number(indicators['bollinger_middle'].iloc[-1], 2),
                        "Bollinger Lower": format_number(indicators['bollinger_lower'].iloc[-1], 2),
                        "Bollinger Width": format_number(indicators['bollinger_width'].iloc[-1], 2),
                        "ATR (14)": format_number(indicators['atr'].iloc[-1], 2)
                    }
                    
                    volatility_table = pd.DataFrame([volatility_data])
                    st.table(volatility_table)
                
                with indicator_tabs[3]:
                    # Support/Resistance
                    st.markdown("#### Support & Resistance Levels")
                    
                    # Calculate support and resistance levels
                    resistance_levels = []
                    support_levels = []
                    
                    # Simple pivot point calculation
                    last_data = stock_data.iloc[-1]
                    pivot = (last_data['High'] + last_data['Low'] + last_data['Close']) / 3
                    
                    s1 = (2 * pivot) - last_data['High']
                    s2 = pivot - (last_data['High'] - last_data['Low'])
                    r1 = (2 * pivot) - last_data['Low']
                    r2 = pivot + (last_data['High'] - last_data['Low'])
                    
                    support_levels = [s1, s2]
                    resistance_levels = [r1, r2]
                    
                    # Display levels
                    support_resistance_data = {
                        "Pivot Point": format_number(pivot, 2),
                        "Resistance 1": format_number(r1, 2),
                        "Resistance 2": format_number(r2, 2),
                        "Support 1": format_number(s1, 2),
                        "Support 2": format_number(s2, 2)
                    }
                    
                    sr_table = pd.DataFrame([support_resistance_data])
                    st.table(sr_table)
                
                # Add indicators disclaimer
                st.warning("""
                **Disclaimer**: Technical indicators are tools used by traders to make decisions. They are not guaranteed predictors of future price movements.
                Use these indicators as part of a comprehensive trading strategy and be aware of their limitations.
                """)

    elif sidebar_section == "💼 Portfolio Tracker":
        # -------- Portfolio Tracker Section --------
        st.markdown("<div class='main-header'>Portfolio Tracker</div>", unsafe_allow_html=True)
        
        # Get current user portfolio or create a new one
        portfolio = st.session_state.portfolio
        
        # Create tabs for different portfolio sections
        portfolio_tabs = st.tabs(["Portfolio Overview", "Add/Remove Stocks", "Transaction History"])
        
        with portfolio_tabs[0]:
            # Portfolio Overview tab
            st.markdown("### Portfolio Summary")
            
            if portfolio.is_empty():
                st.info("Your portfolio is empty. Add stocks in the 'Add/Remove Stocks' tab.")
            else:
                # Display portfolio data
                portfolio_df = portfolio.to_dataframe()
                
                # Calculate current prices and update portfolio
                current_prices = portfolio.get_current_prices()
                
                # Display portfolio value and metrics
                total_value = portfolio.get_total_value()
                total_cost = sum(row['Quantity'] * row['Avg Price'] for _, row in portfolio_df.iterrows())
                total_gain_loss = total_value - total_cost
                total_gain_loss_pct = (total_gain_loss / total_cost) * 100 if total_cost > 0 else 0
                
                # Display metrics in columns
                metric_cols = st.columns(3)
                
                with metric_cols[0]:
                    st.metric(
                        label="Total Portfolio Value",
                        value=f"${format_number(total_value, 2)}"
                    )
                
                with metric_cols[1]:
                    st.metric(
                        label="Total Gain/Loss",
                        value=f"${format_number(total_gain_loss, 2)}",
                        delta=f"{format_number(total_gain_loss_pct, 2)}%"
                    )
                
                with metric_cols[2]:
                    # Count number of positions
                    num_positions = len(portfolio_df)
                    st.metric(
                        label="Number of Positions",
                        value=num_positions
                    )
                
                # Display holdings table
                st.markdown("### Holdings")
                
                # Enhance the dataframe with current prices and gain/loss
                enhanced_df = portfolio_df.copy()
                
                # Add current price and calculated fields
                enhanced_df['Current Price'] = enhanced_df['Symbol'].map(lambda symbol: current_prices.get(symbol, 0))
                enhanced_df['Current Value'] = enhanced_df['Quantity'] * enhanced_df['Current Price']
                enhanced_df['Cost Basis'] = enhanced_df['Quantity'] * enhanced_df['Avg Price']
                enhanced_df['Gain/Loss $'] = enhanced_df['Current Value'] - enhanced_df['Cost Basis']
                enhanced_df['Gain/Loss %'] = (enhanced_df['Gain/Loss $'] / enhanced_df['Cost Basis']) * 100
                
                # Format the dataframe for display
                display_df = pd.DataFrame({
                    'Symbol': enhanced_df['Symbol'],
                    'Quantity': enhanced_df['Quantity'],
                    'Avg Price': enhanced_df['Avg Price'].map(lambda x: f"${format_number(x, 2)}"),
                    'Current Price': enhanced_df['Current Price'].map(lambda x: f"${format_number(x, 2)}"),
                    'Current Value': enhanced_df['Current Value'].map(lambda x: f"${format_number(x, 2)}"),
                    'Gain/Loss': enhanced_df.apply(
                        lambda row: f"${format_number(row['Gain/Loss $'], 2)} ({format_number(row['Gain/Loss %'], 2)}%)",
                        axis=1
                    )
                })
                
                st.dataframe(display_df, use_container_width=True)
                
                # Display portfolio composition chart
                st.markdown("### Portfolio Composition")
                
                if not portfolio.is_empty():
                    composition = portfolio.get_portfolio_composition()
                    
                    # Create pie chart
                    fig = go.Figure(data=[go.Pie(
                        labels=list(composition.keys()),
                        values=list(composition.values()),
                        hole=.4,
                        marker_colors=['#1E88E5', '#42A5F5', '#90CAF9', '#E3F2FD', '#0D47A1', '#5C6BC0', '#3949AB']
                    )])
                    
                    fig.update_layout(
                        title="Portfolio Allocation by Stock",
                        height=400,
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                # Performance chart
                st.markdown("### Performance")
                st.info("Portfolio performance tracking will be available in a future update.")

        with portfolio_tabs[1]:
            # Add/Remove Stocks tab
            st.markdown("### Manage Portfolio")
            
            # Create columns for add and remove forms
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Add Stock")
                
                with st.form("add_stock_form"):
                    add_symbol = st.text_input("Symbol", key="add_symbol").upper()
                    add_quantity = st.number_input("Quantity", min_value=0.01, value=1.0, step=1.0, key="add_quantity")
                    add_price = st.number_input("Purchase Price per Share ($)", min_value=0.01, value=100.0, step=0.01, key="add_price")
                    
                    add_submit = st.form_submit_button("Add Stock")
                
                if add_submit and add_symbol and add_quantity > 0 and add_price > 0:
                    try:
                        # Verify the symbol exists
                        test_data = load_stock_data(add_symbol, period="1d")
                        
                        if test_data.empty:
                            st.error(f"Symbol {add_symbol} not found. Please check the symbol and try again.")
                        else:
                            # Add to portfolio
                            portfolio.add_stock(add_symbol, add_quantity, add_price)
                            st.success(f"Added {add_quantity} shares of {add_symbol} to portfolio.")
                            # Rerun to update the UI
                            st.rerun()
                    except Exception as e:
                        st.error(f"Error adding stock: {str(e)}")
            
            with col2:
                st.markdown("#### Remove Stock")
                
                if portfolio.is_empty():
                    st.info("Your portfolio is empty. Add stocks first.")
                else:
                    portfolio_df = portfolio.to_dataframe()
                    symbols = portfolio_df['Symbol'].tolist()
                    
                    with st.form("remove_stock_form"):
                        remove_symbol = st.selectbox("Symbol", symbols, key="remove_symbol")
                        
                        # Get max quantity for the selected symbol
                        max_quantity = portfolio_df[portfolio_df['Symbol'] == remove_symbol]['Quantity'].iloc[0] if remove_symbol in portfolio_df['Symbol'].values else 0
                        
                        remove_quantity = st.number_input("Quantity to Sell", min_value=0.01, max_value=float(max_quantity), value=min(1.0, float(max_quantity)), step=0.01, key="remove_quantity")
                        remove_price = st.number_input("Selling Price per Share ($)", min_value=0.01, value=100.0, step=0.01, key="remove_price")
                        
                        remove_submit = st.form_submit_button("Sell Stock")
                    
                    if remove_submit and remove_symbol and remove_quantity > 0 and remove_price > 0:
                        try:
                            # Remove from portfolio
                            success = portfolio.remove_stock(remove_symbol, remove_quantity, remove_price)
                            
                            if success:
                                st.success(f"Sold {remove_quantity} shares of {remove_symbol} from portfolio.")
                                # Rerun to update the UI
                                st.rerun()
                            else:
                                st.error(f"Failed to sell {remove_symbol}. Check quantity and try again.")
                        except Exception as e:
                            st.error(f"Error selling stock: {str(e)}")

        with portfolio_tabs[2]:
            # Transaction History tab
            st.markdown("### Transaction History")
            
            if portfolio.is_empty() and not portfolio.get_transaction_history().size > 0:
                st.info("No transactions yet. Add or remove stocks to create a transaction history.")
            else:
                # Get transaction history
                transactions = portfolio.get_transaction_history()
                
                if transactions.empty:
                    st.info("No transactions recorded yet.")
                else:
                    # Format for display
                    transactions['Total'] = transactions['Price'] * transactions['Quantity']
                    
                    display_transactions = pd.DataFrame({
                        'Date': transactions['Timestamp'].dt.strftime('%Y-%m-%d %H:%M'),
                        'Symbol': transactions['Symbol'],
                        'Action': transactions['Action'].str.capitalize(),
                        'Quantity': transactions['Quantity'],
                        'Price': transactions['Price'].map(lambda x: f"${format_number(x, 2)}"),
                        'Total Value': transactions['Total'].map(lambda x: f"${format_number(x, 2)}")
                    })
                    
                    st.dataframe(display_transactions, use_container_width=True)

    elif sidebar_section == "🔎 Stock Comparison":
        # -------- Stock Comparison Section --------
        st.markdown("<div class='main-header'>Stock Comparison</div>", unsafe_allow_html=True)
        
        st.markdown("""
        <div class='info-text'>
        Compare multiple stocks side by side. Enter stock tickers to add them to the comparison.
        </div>
        """, unsafe_allow_html=True)
        
        # Add ticker input
        col1, col2 = st.columns([3, 1])
        
        with col1:
            new_ticker = st.text_input("Enter Stock Ticker Symbol:", key="compare_ticker").upper()
        
        with col2:
            period = st.selectbox(
                "Select Period:",
                ["1mo", "3mo", "6mo", "1y", "2y", "5y"],
                index=3
            )
        
        # Add button to add to comparison list
        if st.button("Add to Comparison") and new_ticker:
            if new_ticker not in st.session_state.comparison_tickers:
                # Verify the ticker exists
                test_data = load_stock_data(new_ticker, period="1d")
                
                if test_data.empty:
                    st.error(f"Symbol {new_ticker} not found. Please check the symbol and try again.")
                else:
                    st.session_state.comparison_tickers.append(new_ticker)
                    st.success(f"Added {new_ticker} to comparison.")
                    st.rerun()
            else:
                st.warning(f"{new_ticker} is already in the comparison list.")
        
        # Display current comparison list
        st.markdown("### Stocks to Compare")
        
        if not st.session_state.comparison_tickers:
            st.info("No stocks added to comparison yet. Add stocks using the form above.")
        else:
            # Display as chips with remove option
            ticker_cols = st.columns(min(4, len(st.session_state.comparison_tickers)))
            
            for i, ticker in enumerate(st.session_state.comparison_tickers):
                with ticker_cols[i % 4]:
                    st.markdown(f"""
                    <div style="background-color: #e0e0e0; border-radius: 16px; padding: 8px 12px; margin: 4px; display: inline-block;">
                        {ticker}
                    </div>
                    """, unsafe_allow_html=True)
            
            # Button to clear all
            if st.button("Clear All"):
                st.session_state.comparison_tickers = []
                st.rerun()
            
            # Button to remove selected
            ticker_to_remove = st.selectbox("Select ticker to remove:", [""] + st.session_state.comparison_tickers)
            if st.button("Remove Selected") and ticker_to_remove:
                st.session_state.comparison_tickers.remove(ticker_to_remove)
                st.rerun()
        
        # Display comparison chart
        if len(st.session_state.comparison_tickers) > 0:
            st.markdown("### Price Comparison Chart (Normalized)")
            
            # Load data for each ticker
            data_dict = {}
            
            with st.spinner("Loading stock data..."):
                for ticker in st.session_state.comparison_tickers:
                    data = load_stock_data(ticker, period=period)
                    if not data.empty:
                        data_dict[ticker] = data
            
            if data_dict:
                st.plotly_chart(plot_comparison_chart(data_dict), use_container_width=True)
            
            # Display comparison metrics
            if len(data_dict) > 1:
                st.markdown("### Performance Metrics Comparison")
                
                # Calculate key metrics for each stock
                metrics_data = []
                
                for ticker, data in data_dict.items():
                    if data.empty:
                        continue
                    
                    # Calculate metrics
                    current_price = data['Close'].iloc[-1]
                    start_price = data['Close'].iloc[0]
                    change_pct = ((current_price - start_price) / start_price) * 100
                    
                    # Weekly volatility (standard deviation of daily returns)
                    returns = data['Close'].pct_change().dropna()
                    weekly_volatility = returns.std() * 100
                    
                    # 50-day moving average
                    ma_50 = data['Close'].rolling(window=min(50, len(data))).mean().iloc[-1]
                    ma_position = "Above MA" if current_price > ma_50 else "Below MA"
                    
                    # Add to metrics data
                    metrics_data.append({
                        'Ticker': ticker,
                        'Current Price': f"${format_number(current_price, 2)}",
                        'Period Change': f"{format_number(change_pct, 2)}%",
                        'Volatility': f"{format_number(weekly_volatility, 2)}%",
                        'MA Position': ma_position
                    })
                
                # Display metrics table
                if metrics_data:
                    metrics_df = pd.DataFrame(metrics_data)
                    st.dataframe(metrics_df.set_index('Ticker'), use_container_width=True)
                
                # Calculate and display correlation matrix
                st.markdown("### Correlation Matrix")
                
                price_data = {}
                for ticker, data in data_dict.items():
                    if not data.empty:
                        price_data[ticker] = data['Close']
                
                if len(price_data) > 1:
                    # Create DataFrame with all close prices
                    corr_df = pd.DataFrame(price_data)
                    
                    # Calculate correlation matrix
                    correlation_matrix = corr_df.corr()
                    
                    # Create heatmap
                    fig = go.Figure(data=go.Heatmap(
                        z=correlation_matrix.values,
                        x=correlation_matrix.columns,
                        y=correlation_matrix.index,
                        colorscale='Blues',
                        zmin=-1,
                        zmax=1
                    ))
                    
                    fig.update_layout(
                        title="Stock Price Correlation Matrix",
                        height=400,
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.markdown("""
                    #### Correlation Analysis
                    
                    High correlation between stocks reduces diversification benefits in a portfolio. For better diversification, consider including stocks with lower correlation.
                    """)

# Add footer to the app
st.markdown("""
<div style="text-align: center; margin-top: 2rem; padding-top: 1rem; border-top: 1px solid #e0e0e0;">
    <p>Stock Market AI Predictor © 2023. All data is for informational purposes only.</p>
    <p style="font-size: 0.8rem;">Powered by Streamlit, yfinance, and Prophet.</p>
</div>
""", unsafe_allow_html=True)