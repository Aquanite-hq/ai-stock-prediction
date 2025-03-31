import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
from datetime import datetime, timedelta

@st.cache_data(ttl=3600)  # Cache data for 1 hour
def load_stock_data(ticker, period="1y"):
    """
    Load stock price data using yfinance
    
    Args:
        ticker (str): Stock ticker symbol
        period (str): Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
        
    Returns:
        DataFrame: OHLCV data for the stock
    """
    try:
        # Check if ticker is valid
        if not ticker or not isinstance(ticker, str):
            st.warning("Please enter a valid stock ticker.")
            return pd.DataFrame()
        
        # Normalize ticker
        ticker = ticker.strip().upper()
        
        # Use a default period if the provided one is invalid
        valid_periods = ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"]
        if period not in valid_periods:
            st.warning(f"Invalid period '{period}'. Using default period of '1y'.")
            period = "1y"
        
        # Try to fetch data with error handling
        stock = yf.Ticker(ticker)
        data = stock.history(period=period)
        
        if data.empty:
            st.warning(f"No data found for {ticker}. The ticker symbol may be invalid or there may be no data for the selected period.")
            return pd.DataFrame()
        
        # Check if we got meaningful data
        if len(data) < 5:  # Arbitrary threshold, but we need enough data for analysis
            st.warning(f"Not enough data points for {ticker} in the selected period.")
            return pd.DataFrame()
            
        return data
    except Exception as e:
        st.error(f"Error loading data for {ticker}: {str(e)}")
        st.info("Please check if the ticker symbol is valid and try again.")
        return pd.DataFrame()

@st.cache_data(ttl=3600)  # Cache data for 1 hour
def get_company_info(ticker):
    """
    Get company information for a given ticker
    
    Args:
        ticker (str): Stock ticker symbol
        
    Returns:
        dict: Company information
    """
    try:
        # Check if ticker is valid
        if not ticker or not isinstance(ticker, str):
            st.warning("Please enter a valid stock ticker.")
            return {}
        
        # Normalize ticker
        ticker = ticker.strip().upper()
        
        stock = yf.Ticker(ticker)
        
        try:
            info = stock.info
            if not info or len(info) < 5:  # Basic check if we got meaningful data
                st.warning(f"Could not retrieve information for {ticker}. The ticker symbol may be invalid.")
                return {
                    'name': ticker,
                    'sector': 'N/A',
                    'industry': 'N/A',
                    'website': 'N/A',
                    'market_cap': 0,
                    'pe_ratio': 0,
                    'dividend_yield': 0,
                    'description': 'Information not available for this ticker.'
                }
                
            # Extract relevant information
            company_info = {
                'name': info.get('shortName', ticker),
                'sector': info.get('sector', 'N/A'),
                'industry': info.get('industry', 'N/A'),
                'website': info.get('website', 'N/A'),
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('trailingPE', 0),
                'dividend_yield': info.get('dividendYield', 0),
                'description': info.get('longBusinessSummary', 'No description available.')
            }
            
            return company_info
        except AttributeError:
            st.warning(f"Could not retrieve company information for {ticker}.")
            return {
                'name': ticker,
                'sector': 'N/A',
                'industry': 'N/A',
                'description': 'Information not available.'
            }
            
    except Exception as e:
        st.error(f"Error getting company info for {ticker}: {str(e)}")
        return {
            'name': ticker,
            'description': 'Error retrieving information.'
        }



@st.cache_data(ttl=3600)  # Cache data for 1 hour
def get_market_indices():
    """
    Get data for major market indices
    
    Returns:
        DataFrame: Data for major indices
    """
    try:
        # List of major indices
        indices = ['^GSPC', '^DJI', '^IXIC', '^RUT']
        names = ['S&P 500', 'Dow Jones', 'NASDAQ', 'Russell 2000']
        
        data = []
        errors = []
        
        for idx, index_ticker in enumerate(indices):
            try:
                index_data = yf.Ticker(index_ticker)
                history = index_data.history(period="1d")
                
                if not history.empty and 'Close' in history.columns:
                    last_close = history['Close'].iloc[-1]
                    prev_close = history['Close'].iloc[-2] if len(history) > 1 else last_close
                    change = last_close - prev_close
                    change_pct = (change / prev_close) * 100 if prev_close != 0 else 0
                    
                    data.append({
                        'Index': names[idx],
                        'Price': last_close,
                        'Change': change,
                        'Change %': change_pct
                    })
                else:
                    # Add a fallback entry with no change data
                    data.append({
                        'Index': names[idx],
                        'Price': 0,
                        'Change': 0,
                        'Change %': 0
                    })
                    errors.append(f"Could not retrieve data for {names[idx]}")
            except Exception as e:
                # Add a fallback entry in case of error for this specific index
                data.append({
                    'Index': names[idx],
                    'Price': 0,
                    'Change': 0,
                    'Change %': 0
                })
                errors.append(f"Error with {names[idx]}: {str(e)}")
        
        # If we have errors but still got some data, show a warning
        if errors and data:
            st.warning(f"Some market data could not be retrieved: {', '.join(errors)}")
            
        # If we got no data at all, that's an error
        if not data:
            st.error("Could not retrieve any market indices data.")
            # Return empty DataFrame with expected columns
            return pd.DataFrame(columns=['Index', 'Price', 'Change', 'Change %'])
            
        return pd.DataFrame(data)
    except Exception as e:
        st.error(f"Error getting market indices: {str(e)}")
        # Return empty DataFrame with expected columns
        return pd.DataFrame(columns=['Index', 'Price', 'Change', 'Change %'])

@st.cache_data(ttl=3600)  # Cache data for 1 hour
def get_stock_fundamentals(ticker):
    """
    Get fundamental data for a stock
    
    Args:
        ticker (str): Stock ticker symbol
        
    Returns:
        dict: Fundamental data
    """
    try:
        stock = yf.Ticker(ticker)
        
        # Get financials
        income_stmt = stock.financials
        balance_sheet = stock.balance_sheet
        cash_flow = stock.cashflow
        
        # Return fundamental data
        return {
            'income_statement': income_stmt,
            'balance_sheet': balance_sheet,
            'cash_flow': cash_flow
        }
    except Exception as e:
        st.error(f"Error getting fundamentals for {ticker}: {e}")
        return {}