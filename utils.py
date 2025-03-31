import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timedelta
import pytz
from pandas.tseries.holiday import USFederalHolidayCalendar

def format_number(num, decimal_places=2):
    """
    Format a number with comma as thousands separator
    
    Args:
        num (float): Number to format
        decimal_places (int): Number of decimal places
        
    Returns:
        str: Formatted number
    """
    if not isinstance(num, (int, float)):
        return "N/A"
    
    if num is None:
        return "N/A"
    
    return f"{num:,.{decimal_places}f}"

def calculate_percentage_change(old_value, new_value):
    """
    Calculate percentage change between two values
    
    Args:
        old_value (float): Old value
        new_value (float): New value
        
    Returns:
        float: Percentage change
    """
    if old_value == 0:
        return float('inf') if new_value > 0 else float('-inf') if new_value < 0 else 0
    
    return ((new_value - old_value) / old_value) * 100

def get_market_status():
    """
    Check if the US stock market is currently open
    
    Returns:
        str: Market status (Open/Closed)
    """
    # Get current time in US Eastern Time Zone
    eastern = pytz.timezone('US/Eastern')
    now = datetime.now(eastern)
    
    # Get day of the week (0 = Monday, 6 = Sunday)
    day_of_week = now.weekday()
    
    # Check if it's a weekend
    if day_of_week >= 5:  # Saturday or Sunday
        return "Closed"
    
    # Check if it's between 9:30 AM and 4:00 PM ET
    market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
    
    # Check if it's a US market holiday
    if is_market_holiday(now.date()):
        return "Closed (Holiday)"
    
    if market_open <= now <= market_close:
        return "Open"
    elif now < market_open:
        return f"Closed (Opens in {market_open - now})"
    else:
        next_open = market_open + timedelta(days=1)
        # Skip to Monday if it's Friday after market close
        if day_of_week == 4:  # Friday
            next_open = next_open + timedelta(days=2)
            
        return f"Closed (Opens in {next_open - now})"

def is_market_holiday(date):
    """
    Check if a given date is a US market holiday
    
    Args:
        date (date): Date to check
        
    Returns:
        bool: True if it's a holiday, False otherwise
    """
    cal = USFederalHolidayCalendar()
    holidays = cal.holidays(start=date, end=date)
    return len(holidays) > 0

def get_market_holidays(year=None):
    """
    Get the market holidays for a specific year
    
    Args:
        year (int): Year to get holidays for, defaults to current year
        
    Returns:
        list: List of holiday dates
    """
    if year is None:
        year = datetime.now().year
        
    cal = USFederalHolidayCalendar()
    start_date = pd.Timestamp(f'{year}-01-01')
    end_date = pd.Timestamp(f'{year}-12-31')
    
    holidays = cal.holidays(start=start_date, end=end_date)
    return holidays

def get_trading_days(start_date, end_date):
    """
    Get the number of trading days between two dates
    
    Args:
        start_date (datetime): Start date
        end_date (datetime): End date
        
    Returns:
        int: Number of trading days
    """
    # Get business days (Monday to Friday)
    business_days = pd.date_range(start=start_date, end=end_date, freq='B')
    
    # Get holidays
    cal = USFederalHolidayCalendar()
    holidays = cal.holidays(start=start_date, end=end_date)
    
    # Exclude holidays from business days
    trading_days = business_days.difference(holidays)
    
    return len(trading_days)

def calculate_annualized_return(initial_value, final_value, days):
    """
    Calculate annualized return
    
    Args:
        initial_value (float): Initial investment value
        final_value (float): Final investment value
        days (int): Number of days for the investment period
        
    Returns:
        float: Annualized return as a percentage
    """
    if initial_value <= 0 or days <= 0:
        return 0
    
    # Calculate total return
    total_return = (final_value / initial_value) - 1
    
    # Annualize the return
    annualized_return = ((1 + total_return) ** (365 / days)) - 1
    
    return annualized_return * 100

def calculate_sharpe_ratio(returns, risk_free_rate=0.03):
    """
    Calculate Sharpe ratio
    
    Args:
        returns (Series): Daily returns
        risk_free_rate (float): Risk-free rate (annual)
        
    Returns:
        float: Sharpe ratio
    """
    # Convert annual risk-free rate to daily
    daily_risk_free = ((1 + risk_free_rate) ** (1/252)) - 1
    
    # Calculate excess returns
    excess_returns = returns - daily_risk_free
    
    # Calculate Sharpe ratio
    sharpe_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(252)
    
    return sharpe_ratio

def calculate_drawdown(equity_curve):
    """
    Calculate drawdown
    
    Args:
        equity_curve (Series): Equity curve (portfolio value over time)
        
    Returns:
        tuple: (Drawdown series, Maximum drawdown)
    """
    # Calculate running maximum
    running_max = equity_curve.cummax()
    
    # Calculate drawdown
    drawdown = (equity_curve / running_max) - 1
    
    # Calculate maximum drawdown
    max_drawdown = drawdown.min()
    
    return drawdown, max_drawdown