import pandas as pd
import numpy as np
import streamlit as st
import yfinance as yf
from datetime import datetime
from db_service import DatabaseService

class Portfolio:
    """Portfolio class to manage stock portfolio"""
    
    def __init__(self, db_portfolio_id=None):
        """
        Initialize portfolio
        
        Args:
            db_portfolio_id (int, optional): Database portfolio ID to load from database
        """
        self.db_service = DatabaseService()
        
        # Memory cache for holdings and transactions
        self.holdings = {}  # Dictionary to track holdings: {symbol: {'quantity': int, 'avg_price': float}}
        self.transactions = []  # List to track all transactions
        
        # Load from database if ID is provided
        if db_portfolio_id:
            self.load_from_database(db_portfolio_id)
        else:
            # Get the first portfolio or create one if none exists
            portfolios = self.db_service.get_user_portfolios()
            if portfolios:
                self.portfolio_id = portfolios[0].id
                self.load_from_database(self.portfolio_id)
            else:
                # Create a default portfolio
                portfolio = self.db_service.create_portfolio("My Portfolio")
                self.portfolio_id = portfolio.id
    
    def load_from_database(self, portfolio_id):
        """Load portfolio data from database"""
        self.portfolio_id = portfolio_id
        
        # Load holdings
        holdings = self.db_service.get_portfolio_holdings(portfolio_id)
        for holding in holdings:
            self.holdings[holding.symbol] = {
                'quantity': holding.quantity,
                'avg_price': holding.average_price
            }
        
        # Load transactions
        transactions = self.db_service.get_portfolio_transactions(portfolio_id)
        for transaction in transactions:
            self.transactions.append({
                'date': transaction.timestamp,
                'symbol': transaction.symbol,
                'action': transaction.action,
                'quantity': transaction.quantity,
                'price': transaction.price,
                'total': transaction.quantity * transaction.price
            })
    
    def add_stock(self, symbol, quantity, price):
        """
        Add a stock to the portfolio
        
        Args:
            symbol (str): Stock ticker symbol
            quantity (int): Number of shares
            price (float): Purchase price per share
        """
        symbol = symbol.upper()
        
        # Add to database
        self.db_service.add_holding(self.portfolio_id, symbol, quantity, price)
        
        # Update in-memory cache
        if symbol in self.holdings:
            # Calculate new average price
            current_quantity = self.holdings[symbol]['quantity']
            current_avg_price = self.holdings[symbol]['avg_price']
            
            new_quantity = current_quantity + quantity
            new_avg_price = ((current_quantity * current_avg_price) + (quantity * price)) / new_quantity
            
            self.holdings[symbol] = {
                'quantity': new_quantity,
                'avg_price': new_avg_price
            }
        else:
            self.holdings[symbol] = {
                'quantity': quantity,
                'avg_price': price
            }
        
        # Record transaction in memory
        self.transactions.append({
            'date': datetime.now(),
            'symbol': symbol,
            'action': 'buy',
            'quantity': quantity,
            'price': price,
            'total': quantity * price
        })
    
    def remove_stock(self, symbol, quantity, price):
        """
        Remove a stock from the portfolio (sell)
        
        Args:
            symbol (str): Stock ticker symbol
            quantity (int): Number of shares to sell
            price (float): Selling price per share
            
        Returns:
            bool: True if successful, False otherwise
        """
        symbol = symbol.upper()
        
        if symbol not in self.holdings:
            return False
        
        if quantity > self.holdings[symbol]['quantity']:
            return False
        
        # Remove from database
        result = self.db_service.remove_holding(self.portfolio_id, symbol, quantity, price)
        
        if result:
            # Update in-memory cache
            self.holdings[symbol]['quantity'] -= quantity
            
            # If quantity is zero, remove the stock from holdings
            if self.holdings[symbol]['quantity'] == 0:
                del self.holdings[symbol]
            
            # Record transaction in memory
            self.transactions.append({
                'date': datetime.now(),
                'symbol': symbol,
                'action': 'sell',
                'quantity': quantity,
                'price': price,
                'total': quantity * price
            })
            
            return True
        
        return False
    
    def get_current_prices(self):
        """
        Get current prices for all holdings
        
        Returns:
            dict: Dictionary mapping symbols to current prices
        """
        prices = {}
        
        for symbol in self.holdings.keys():
            try:
                data = yf.Ticker(symbol).history(period='1d')
                if not data.empty:
                    prices[symbol] = data['Close'].iloc[-1]
                else:
                    prices[symbol] = 0
            except Exception:
                prices[symbol] = 0
        
        return prices
    
    def get_total_value(self):
        """
        Calculate total portfolio value
        
        Returns:
            float: Total portfolio value
        """
        total_value = 0
        current_prices = self.get_current_prices()
        
        for symbol, details in self.holdings.items():
            if symbol in current_prices:
                total_value += details['quantity'] * current_prices[symbol]
        
        return total_value
    
    def get_total_gain_loss(self):
        """
        Calculate total gain/loss
        
        Returns:
            float: Total gain/loss in dollars
        """
        total_gain_loss = 0
        current_prices = self.get_current_prices()
        
        for symbol, details in self.holdings.items():
            if symbol in current_prices:
                current_value = details['quantity'] * current_prices[symbol]
                cost_basis = details['quantity'] * details['avg_price']
                total_gain_loss += current_value - cost_basis
        
        return total_gain_loss
    
    def to_dataframe(self):
        """
        Convert portfolio to DataFrame
        
        Returns:
            DataFrame: Portfolio holdings
        """
        if not self.holdings:
            return pd.DataFrame()
        
        data = []
        current_prices = self.get_current_prices()
        
        for symbol, details in self.holdings.items():
            quantity = details['quantity']
            avg_price = details['avg_price']
            current_price = current_prices.get(symbol, 0)
            
            cost_basis = quantity * avg_price
            current_value = quantity * current_price
            gain_loss = current_value - cost_basis
            gain_loss_percentage = (gain_loss / cost_basis) * 100 if cost_basis != 0 else 0
            
            data.append({
                'Symbol': symbol,
                'Quantity': quantity,
                'Average Price': avg_price,
                'Current Price': current_price,
                'Cost Basis': cost_basis,
                'Current Value': current_value,
                'Gain/Loss': gain_loss,
                'Gain/Loss %': gain_loss_percentage
            })
        
        return pd.DataFrame(data)
    
    def get_transaction_history(self):
        """
        Get transaction history
        
        Returns:
            DataFrame: Transaction history
        """
        return pd.DataFrame(self.transactions)
    
    def is_empty(self):
        """Check if portfolio is empty"""
        return len(self.holdings) == 0
    
    def get_portfolio_composition(self):
        """
        Get portfolio composition by percentage
        
        Returns:
            dict: Dictionary mapping symbols to percentage of portfolio
        """
        composition = {}
        total_value = self.get_total_value()
        current_prices = self.get_current_prices()
        
        if total_value == 0:
            return {}
        
        for symbol, details in self.holdings.items():
            if symbol in current_prices:
                value = details['quantity'] * current_prices[symbol]
                composition[symbol] = (value / total_value) * 100
        
        return composition
