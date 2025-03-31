import pandas as pd
import numpy as np
import streamlit as st

def calculate_technical_indicators(data):
    """
    Calculate technical indicators for a given stock data
    
    Args:
        data (DataFrame): Historical stock price data with OHLCV
        
    Returns:
        dict: Dictionary with technical indicators
    """
    indicators = {}
    
    # Simple Moving Averages
    indicators['sma20'] = calculate_sma(data, 20)
    indicators['sma50'] = calculate_sma(data, 50)
    indicators['sma200'] = calculate_sma(data, 200)
    
    # Exponential Moving Averages
    indicators['ema12'] = calculate_ema(data, 12)
    indicators['ema26'] = calculate_ema(data, 26)
    
    # MACD
    macd_data = calculate_macd(data)
    indicators['macd'] = macd_data['macd']
    indicators['macd_signal'] = macd_data['signal']
    indicators['macd_histogram'] = macd_data['histogram']
    
    # RSI
    indicators['rsi'] = calculate_rsi(data)
    
    # Bollinger Bands
    bollinger = calculate_bollinger_bands(data)
    indicators['bollinger_upper'] = bollinger['upper']
    indicators['bollinger_middle'] = bollinger['middle']
    indicators['bollinger_lower'] = bollinger['lower']
    # Calculate bollinger width (upper - lower) / middle
    indicators['bollinger_width'] = (bollinger['upper'] - bollinger['lower']) / bollinger['middle']
    
    # Stochastic Oscillator
    stochastic = calculate_stochastic_oscillator(data)
    indicators['stoch_k'] = stochastic['k']
    indicators['stoch_d'] = stochastic['d']
    
    # On Balance Volume
    indicators['obv'] = calculate_obv(data)
    
    # Average True Range
    indicators['atr'] = calculate_atr(data)
    
    # Generate trading signals
    close_price = data['Close'].iloc[-1] if not data.empty else None
    indicators['signals'] = get_indicator_signals(indicators, close_price)
    
    return indicators

def calculate_sma(data, window):
    """Calculate Simple Moving Average"""
    return data['Close'].rolling(window=window).mean()

def calculate_ema(data, span):
    """Calculate Exponential Moving Average"""
    return data['Close'].ewm(span=span, adjust=False).mean()

def calculate_macd(data, fast=12, slow=26, signal=9):
    """Calculate MACD, Signal Line, and Histogram"""
    ema_fast = calculate_ema(data, fast)
    ema_slow = calculate_ema(data, slow)
    
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    macd_histogram = macd - macd_signal
    
    return {
        'macd': macd,
        'signal': macd_signal,
        'histogram': macd_histogram
    }

def calculate_rsi(data, window=14):
    """Calculate Relative Strength Index"""
    delta = data['Close'].diff()
    
    # Make two series: one for lower closes and one for higher closes
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    
    # Calculate the EWMA
    ma_up = up.ewm(com=window-1, adjust=True, min_periods=window).mean()
    ma_down = down.ewm(com=window-1, adjust=True, min_periods=window).mean()
    
    # Calculate RS (Relative Strength)
    rs = ma_up / ma_down
    
    # Calculate RSI
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def calculate_bollinger_bands(data, window=20, num_std=2):
    """Calculate Bollinger Bands"""
    middle_band = calculate_sma(data, window)
    std_dev = data['Close'].rolling(window=window).std()
    
    upper_band = middle_band + (std_dev * num_std)
    lower_band = middle_band - (std_dev * num_std)
    
    return {
        'upper': upper_band,
        'middle': middle_band,
        'lower': lower_band
    }

def calculate_stochastic_oscillator(data, k_window=14, d_window=3):
    """Calculate Stochastic Oscillator"""
    # Calculate %K
    low_min = data['Low'].rolling(window=k_window).min()
    high_max = data['High'].rolling(window=k_window).max()
    
    k = 100 * ((data['Close'] - low_min) / (high_max - low_min))
    
    # Calculate %D
    d = k.rolling(window=d_window).mean()
    
    return {
        'k': k,
        'd': d
    }

def calculate_obv(data):
    """Calculate On-Balance Volume"""
    obv = pd.Series(index=data.index)
    obv.iloc[0] = 0
    
    for i in range(1, len(data)):
        if data['Close'].iloc[i] > data['Close'].iloc[i-1]:
            obv.iloc[i] = obv.iloc[i-1] + data['Volume'].iloc[i]
        elif data['Close'].iloc[i] < data['Close'].iloc[i-1]:
            obv.iloc[i] = obv.iloc[i-1] - data['Volume'].iloc[i]
        else:
            obv.iloc[i] = obv.iloc[i-1]
    
    return obv

def calculate_atr(data, window=14):
    """Calculate Average True Range"""
    high_low = data['High'] - data['Low']
    high_close = (data['High'] - data['Close'].shift()).abs()
    low_close = (data['Low'] - data['Close'].shift()).abs()
    
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    
    atr = true_range.rolling(window=window).mean()
    
    return atr

def get_indicator_signals(indicators, close_price):
    """Generate trading signals based on technical indicators"""
    signals = {}
    
    # Check if we have enough data
    if close_price is None:
        return signals
    
    # SMA signals
    try:
        sma20 = indicators['sma20'].iloc[-1]
        sma50 = indicators['sma50'].iloc[-1]
        sma200 = indicators['sma200'].iloc[-1]
        
        # Price above/below moving averages
        signals['price_above_sma20'] = close_price > sma20
        signals['price_above_sma50'] = close_price > sma50
        signals['price_above_sma200'] = close_price > sma200
        
        # Golden Cross / Death Cross
        signals['golden_cross'] = (sma50 > sma200) and (indicators['sma50'].shift(1).iloc[-1] <= indicators['sma200'].shift(1).iloc[-1])
        signals['death_cross'] = (sma50 < sma200) and (indicators['sma50'].shift(1).iloc[-1] >= indicators['sma200'].shift(1).iloc[-1])
    except (IndexError, KeyError):
        pass
    
    # MACD signals
    try:
        macd = indicators['macd'].iloc[-1]
        macd_signal = indicators['macd_signal'].iloc[-1]
        macd_hist = indicators['macd_histogram'].iloc[-1]
        
        signals['macd_above_signal'] = macd > macd_signal
        signals['macd_crossover'] = (macd > macd_signal) and (indicators['macd'].shift(1).iloc[-1] <= indicators['macd_signal'].shift(1).iloc[-1])
        signals['macd_crossunder'] = (macd < macd_signal) and (indicators['macd'].shift(1).iloc[-1] >= indicators['macd_signal'].shift(1).iloc[-1])
    except (IndexError, KeyError):
        pass
    
    # RSI signals
    try:
        rsi = indicators['rsi'].iloc[-1]
        
        signals['rsi_oversold'] = rsi < 30
        signals['rsi_overbought'] = rsi > 70
    except (IndexError, KeyError):
        pass
    
    # Bollinger Bands signals
    try:
        upper = indicators['bollinger_upper'].iloc[-1]
        lower = indicators['bollinger_lower'].iloc[-1]
        
        signals['price_above_upper_band'] = close_price > upper
        signals['price_below_lower_band'] = close_price < lower
    except (IndexError, KeyError):
        pass
    
    # Stochastic signals
    try:
        k = indicators['stoch_k'].iloc[-1]
        d = indicators['stoch_d'].iloc[-1]
        
        signals['stoch_oversold'] = k < 20 and d < 20
        signals['stoch_overbought'] = k > 80 and d > 80
        signals['stoch_crossover'] = (k > d) and (indicators['stoch_k'].shift(1).iloc[-1] <= indicators['stoch_d'].shift(1).iloc[-1])
        signals['stoch_crossunder'] = (k < d) and (indicators['stoch_k'].shift(1).iloc[-1] >= indicators['stoch_d'].shift(1).iloc[-1])
    except (IndexError, KeyError):
        pass
    
    # Overall signals
    bullish_count = sum(1 for signal, value in signals.items() if 'above' in signal or 'golden' in signal or 'oversold' in signal or 'crossover' in signal and value)
    bearish_count = sum(1 for signal, value in signals.items() if 'below' in signal or 'death' in signal or 'overbought' in signal or 'crossunder' in signal and value)
    
    signals['overall_bullish'] = bullish_count > bearish_count
    signals['overall_bearish'] = bearish_count > bullish_count
    
    return signals