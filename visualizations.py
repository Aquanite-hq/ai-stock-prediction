import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

def plot_stock_chart(data, ticker):
    """
    Plot a candlestick chart for stock data
    
    Args:
        data (DataFrame): Stock price data with OHLCV
        ticker (str): Stock ticker symbol
        
    Returns:
        Figure: Plotly figure object
    """
    # Create empty figure to start with
    fig = go.Figure()
    
    if data is None or data.empty:
        # Return an empty chart with message
        fig.add_annotation(
            text="No data available for this stock",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=20)
        )
        fig.update_layout(
            title=f'No Data for {ticker}',
            height=600
        )
        return fig
    
    # Check if we have all the required columns
    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    missing_columns = [col for col in required_columns if col not in data.columns]
    
    if missing_columns:
        # Return an error chart
        fig.add_annotation(
            text=f"Missing required data columns: {', '.join(missing_columns)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16)
        )
        fig.update_layout(
            title=f'Incomplete Data for {ticker}',
            height=600
        )
        return fig
    
    try:
        # Create figure with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Add candlestick trace
        fig.add_trace(
            go.Candlestick(
                x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                name='Price'
            ),
            secondary_y=False
        )
        
        # Add volume as bar chart on secondary y-axis
        fig.add_trace(
            go.Bar(
                x=data.index,
                y=data['Volume'],
                name='Volume',
                marker_color='rgba(128, 128, 128, 0.5)'
            ),
            secondary_y=True
        )
        
        # Set chart title and labels
        fig.update_layout(
            title=f'{ticker} Stock Price',
            xaxis_title='Date',
            yaxis_title='Price ($)',
            yaxis2_title='Volume',
            xaxis_rangeslider_visible=False,
            template='plotly_white',
            height=600
        )
        
        # Set y-axes titles and colors
        fig.update_yaxes(title_text="Price ($)", secondary_y=False)
        fig.update_yaxes(title_text="Volume", secondary_y=True, showgrid=False)
    
    except Exception as e:
        # If anything goes wrong, return a figure with error message
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error creating chart: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16)
        )
        fig.update_layout(
            title=f'Error Displaying {ticker} Data',
            height=600
        )
    
    return fig

def plot_prediction_chart(historical_data, predicted_data, ticker):
    """
    Plot a chart with historical and predicted stock prices
    
    Args:
        historical_data (DataFrame): Historical stock price data
        predicted_data (DataFrame): Predicted stock price data
        ticker (str): Stock ticker symbol
        
    Returns:
        Figure: Plotly figure object
    """
    # Create an empty figure
    fig = go.Figure()
    
    # Handle empty data case
    if historical_data is None or historical_data.empty:
        fig.add_annotation(
            text="No historical data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=20)
        )
        fig.update_layout(
            title=f'No Historical Data for {ticker}',
            height=500
        )
        return fig
    
    if predicted_data is None or predicted_data.empty:
        fig.add_annotation(
            text="No prediction data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=20)
        )
        fig.update_layout(
            title=f'No Prediction Data for {ticker}',
            height=500
        )
        return fig
    
    # Check for required columns
    if 'Close' not in historical_data.columns:
        fig.add_annotation(
            text="Missing 'Close' price data in historical data",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16)
        )
        fig.update_layout(
            title=f'Incomplete Historical Data for {ticker}',
            height=500
        )
        return fig
    
    if 'ds' not in predicted_data.columns or 'yhat' not in predicted_data.columns:
        fig.add_annotation(
            text="Missing required columns in prediction data",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16)
        )
        fig.update_layout(
            title=f'Incomplete Prediction Data for {ticker}',
            height=500
        )
        return fig
    
    try:
        # Add historical price trace
        fig.add_trace(
            go.Scatter(
                x=historical_data.index,
                y=historical_data['Close'],
                name='Historical',
                line=dict(color='blue')
            )
        )
        
        # Filter prediction dataframe to only future dates
        try:
            future_data = predicted_data[predicted_data['ds'] > historical_data.index[-1]]
            
            # Check if we have any future predictions
            if future_data.empty:
                fig.add_annotation(
                    text="No future predictions available",
                    xref="paper", yref="paper",
                    x=0.8, y=0.2,
                    showarrow=True,
                    arrowhead=1,
                    ax=0,
                    ay=-40
                )
            else:
                # Add prediction trace
                fig.add_trace(
                    go.Scatter(
                        x=future_data['ds'],
                        y=future_data['yhat'],
                        name='Predicted',
                        line=dict(color='red', dash='dash')
                    )
                )
        except Exception as e:
            # If filtering fails, use all prediction data
            fig.add_trace(
                go.Scatter(
                    x=predicted_data['ds'],
                    y=predicted_data['yhat'],
                    name='Predicted',
                    line=dict(color='red', dash='dash')
                )
            )
            fig.add_annotation(
                text=f"Error filtering predictions: {str(e)}",
                xref="paper", yref="paper",
                x=0.8, y=0.1,
                showarrow=False,
                font=dict(size=10)
            )
        
        # Set chart title and labels
        fig.update_layout(
            title=f'{ticker} Stock Price Prediction',
            xaxis_title='Date',
            yaxis_title='Price ($)',
            template='plotly_white',
            height=500
        )
        
    except Exception as e:
        # If anything goes wrong, return error figure
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error creating prediction chart: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16)
        )
        fig.update_layout(
            title=f'Error in {ticker} Prediction Chart',
            height=500
        )
    
    return fig

def plot_indicators_chart(data, indicators):
    """
    Plot technical indicators chart
    
    Args:
        data (DataFrame): Stock price data
        indicators (dict): Dictionary of technical indicators
        
    Returns:
        Figure: Plotly figure object
    """
    if data.empty or not indicators:
        return go.Figure()
    
    # Create subplots for different indicators
    fig = make_subplots(
        rows=4, 
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=('Price and Moving Averages', 'MACD', 'RSI', 'Stochastic Oscillator'),
        row_heights=[0.5, 0.2, 0.15, 0.15]
    )
    
    # Add price and moving averages
    fig.add_trace(
        go.Scatter(
            x=data.index, 
            y=data['Close'],
            name='Close Price',
            line=dict(color='black', width=2)
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=data.index, 
            y=indicators['sma20'],
            name='SMA 20',
            line=dict(color='blue')
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=data.index, 
            y=indicators['sma50'],
            name='SMA 50',
            line=dict(color='green')
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=data.index, 
            y=indicators['sma200'],
            name='SMA 200',
            line=dict(color='red')
        ),
        row=1, col=1
    )
    
    # Add Bollinger Bands
    fig.add_trace(
        go.Scatter(
            x=data.index, 
            y=indicators['bollinger_upper'],
            name='Bollinger Upper',
            line=dict(color='rgba(173, 204, 255, 0.8)')
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=data.index, 
            y=indicators['bollinger_lower'],
            name='Bollinger Lower',
            line=dict(color='rgba(173, 204, 255, 0.8)'),
            fill='tonexty'
        ),
        row=1, col=1
    )
    
    # Add MACD
    fig.add_trace(
        go.Scatter(
            x=data.index, 
            y=indicators['macd'],
            name='MACD',
            line=dict(color='blue')
        ),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=data.index, 
            y=indicators['macd_signal'],
            name='Signal Line',
            line=dict(color='red')
        ),
        row=2, col=1
    )
    
    # Add MACD Histogram as bar chart
    colors = ['green' if val >= 0 else 'red' for val in indicators['macd_histogram']]
    fig.add_trace(
        go.Bar(
            x=data.index, 
            y=indicators['macd_histogram'],
            name='Histogram',
            marker_color=colors
        ),
        row=2, col=1
    )
    
    # Add RSI
    fig.add_trace(
        go.Scatter(
            x=data.index, 
            y=indicators['rsi'],
            name='RSI',
            line=dict(color='purple')
        ),
        row=3, col=1
    )
    
    # Add RSI oversold/overbought lines
    fig.add_trace(
        go.Scatter(
            x=[data.index[0], data.index[-1]],
            y=[30, 30],
            name='Oversold',
            line=dict(color='green', dash='dash')
        ),
        row=3, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=[data.index[0], data.index[-1]],
            y=[70, 70],
            name='Overbought',
            line=dict(color='red', dash='dash')
        ),
        row=3, col=1
    )
    
    # Add Stochastic Oscillator
    fig.add_trace(
        go.Scatter(
            x=data.index, 
            y=indicators['stoch_k'],
            name='%K',
            line=dict(color='blue')
        ),
        row=4, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=data.index, 
            y=indicators['stoch_d'],
            name='%D',
            line=dict(color='red')
        ),
        row=4, col=1
    )
    
    # Add stochastic oversold/overbought lines
    fig.add_trace(
        go.Scatter(
            x=[data.index[0], data.index[-1]],
            y=[20, 20],
            name='Oversold',
            line=dict(color='green', dash='dash'),
            showlegend=False
        ),
        row=4, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=[data.index[0], data.index[-1]],
            y=[80, 80],
            name='Overbought',
            line=dict(color='red', dash='dash'),
            showlegend=False
        ),
        row=4, col=1
    )
    
    # Update layout
    fig.update_layout(
        height=900,
        template='plotly_white',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Update y-axis ranges for RSI and Stochastic
    fig.update_yaxes(range=[0, 100], row=3, col=1)
    fig.update_yaxes(range=[0, 100], row=4, col=1)
    
    return fig

def plot_comparison_chart(data_dict):
    """
    Plot comparison chart for multiple stocks
    
    Args:
        data_dict (dict): Dictionary mapping stock symbols to their dataframes
        
    Returns:
        Figure: Plotly figure object
    """
    if not data_dict:
        return go.Figure()
    
    # Create figure
    fig = go.Figure()
    
    # Get earliest common date across all dataframes
    start_dates = [df.index[0] for df in data_dict.values() if not df.empty]
    if not start_dates:
        return go.Figure()
    
    common_start = max(start_dates)
    
    # Normalize data to start at 100
    for symbol, df in data_dict.items():
        if df.empty or common_start not in df.index:
            continue
            
        # Get starting index for common date
        start_idx = df.index.get_loc(common_start)
        base_value = df['Close'].iloc[start_idx]
        
        # Calculate normalized values
        normalized = (df['Close'] / base_value) * 100
        
        # Add trace
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=normalized,
                name=symbol,
                mode='lines'
            )
        )
    
    # Update layout
    fig.update_layout(
        title='Stock Price Comparison (Normalized to 100)',
        xaxis_title='Date',
        yaxis_title='Normalized Price',
        template='plotly_white',
        height=500,
        yaxis=dict(ticksuffix='%')
    )
    
    return fig

def plot_portfolio_performance(portfolio_data):
    """
    Plot portfolio performance over time
    
    Args:
        portfolio_data (DataFrame): Portfolio value over time
        
    Returns:
        Figure: Plotly figure object
    """
    if portfolio_data.empty:
        return go.Figure()
    
    # Create figure
    fig = go.Figure()
    
    # Add portfolio value trace
    fig.add_trace(
        go.Scatter(
            x=portfolio_data.index,
            y=portfolio_data['Value'],
            name='Portfolio Value',
            fill='tozeroy',
            line=dict(color='rgb(0, 123, 255)')
        )
    )
    
    # Add annotations for important points
    # Calculate high and low points
    max_point = portfolio_data['Value'].max()
    max_date = portfolio_data.loc[portfolio_data['Value'] == max_point].index[0]
    
    min_point = portfolio_data['Value'].min()
    min_date = portfolio_data.loc[portfolio_data['Value'] == min_point].index[0]
    
    # Add annotation for max point
    fig.add_annotation(
        x=max_date,
        y=max_point,
        text=f"High: ${max_point:.2f}",
        showarrow=True,
        arrowhead=1,
        ax=0,
        ay=-40
    )
    
    # Add annotation for min point
    fig.add_annotation(
        x=min_date,
        y=min_point,
        text=f"Low: ${min_point:.2f}",
        showarrow=True,
        arrowhead=1,
        ax=0,
        ay=40
    )
    
    # Calculate overall gain/loss
    start_value = portfolio_data['Value'].iloc[0]
    end_value = portfolio_data['Value'].iloc[-1]
    total_gain = end_value - start_value
    total_gain_pct = (total_gain / start_value) * 100 if start_value != 0 else 0
    
    # Set chart title and labels
    title_text = f"Portfolio Performance<br><sub>Total Gain/Loss: ${total_gain:.2f} ({total_gain_pct:.2f}%)</sub>"
    
    fig.update_layout(
        title=title_text,
        xaxis_title='Date',
        yaxis_title='Portfolio Value ($)',
        template='plotly_white',
        height=500
    )
    
    return fig