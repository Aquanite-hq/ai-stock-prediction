import streamlit as st
from database import get_session, User, Portfolio as DbPortfolio, Holding, Transaction, Watchlist, WatchlistItem
import yfinance as yf
from datetime import datetime

class DatabaseService:
    """Service class to handle database operations"""
    
    def __init__(self, user_id=1):
        """Initialize with a user ID (defaults to 1 for the default user)"""
        self.user_id = user_id
        self.session = None
        try:
            self.session = get_session()
            if self.session is None:
                print("Failed to get database session in DatabaseService")
                st.error("Database connection error. Some features may not work properly.")
        except Exception as e:
            print(f"Error initializing DatabaseService: {e}")
            st.error(f"Database error: {e}")
    
    def get_user_portfolios(self):
        """Get all portfolios for the current user"""
        if self.session is None:
            print("Cannot get portfolios: no database session")
            return []
        try:
            return self.session.query(DbPortfolio).filter(DbPortfolio.user_id == self.user_id).all()
        except Exception as e:
            print(f"Error getting portfolios: {e}")
            return []
    
    def get_portfolio(self, portfolio_id):
        """Get a specific portfolio by ID"""
        if self.session is None:
            print("Cannot get portfolio: no database session")
            return None
        try:
            return self.session.query(DbPortfolio).filter(
                DbPortfolio.id == portfolio_id,
                DbPortfolio.user_id == self.user_id
            ).first()
        except Exception as e:
            print(f"Error getting portfolio: {e}")
            return None
    
    def create_portfolio(self, name):
        """Create a new portfolio"""
        if self.session is None:
            print("Cannot create portfolio: no database session")
            st.error("Database connection error. Unable to create portfolio.")
            return None
        
        try:
            portfolio = DbPortfolio(name=name, user_id=self.user_id)
            self.session.add(portfolio)
            self.session.commit()
            return portfolio
        except Exception as e:
            print(f"Error creating portfolio: {e}")
            st.error(f"Failed to create portfolio: {e}")
            try:
                self.session.rollback()
            except Exception as rollback_error:
                print(f"Error rolling back transaction: {rollback_error}")
            return None
    
    def add_holding(self, portfolio_id, symbol, quantity, price):
        """Add a holding to a portfolio"""
        # Check if holding already exists
        holding = self.session.query(Holding).filter(
            Holding.portfolio_id == portfolio_id,
            Holding.symbol == symbol
        ).first()
        
        if holding:
            # Update existing holding
            new_quantity = holding.quantity + quantity
            new_avg_price = ((holding.quantity * holding.average_price) + (quantity * price)) / new_quantity
            
            holding.quantity = new_quantity
            holding.average_price = new_avg_price
        else:
            # Create new holding
            holding = Holding(
                portfolio_id=portfolio_id,
                symbol=symbol,
                quantity=quantity,
                average_price=price
            )
            self.session.add(holding)
        
        # Add transaction record
        transaction = Transaction(
            portfolio_id=portfolio_id,
            symbol=symbol,
            action='buy',
            quantity=quantity,
            price=price
        )
        self.session.add(transaction)
        
        self.session.commit()
        return holding
    
    def remove_holding(self, portfolio_id, symbol, quantity, price):
        """Remove (sell) a holding from a portfolio"""
        holding = self.session.query(Holding).filter(
            Holding.portfolio_id == portfolio_id,
            Holding.symbol == symbol
        ).first()
        
        if not holding or holding.quantity < quantity:
            return False
        
        # Update holding
        holding.quantity -= quantity
        
        # Remove holding if quantity is zero
        if holding.quantity == 0:
            self.session.delete(holding)
        
        # Add transaction record
        transaction = Transaction(
            portfolio_id=portfolio_id,
            symbol=symbol,
            action='sell',
            quantity=quantity,
            price=price
        )
        self.session.add(transaction)
        
        self.session.commit()
        return True
    
    def get_portfolio_holdings(self, portfolio_id):
        """Get all holdings for a specific portfolio"""
        return self.session.query(Holding).filter(Holding.portfolio_id == portfolio_id).all()
    
    def get_portfolio_transactions(self, portfolio_id):
        """Get all transactions for a specific portfolio"""
        return self.session.query(Transaction).filter(Transaction.portfolio_id == portfolio_id).all()
    
    def get_user_watchlists(self):
        """Get all watchlists for the current user"""
        return self.session.query(Watchlist).filter(Watchlist.user_id == self.user_id).all()
    
    def get_watchlist(self, watchlist_id):
        """Get a specific watchlist by ID"""
        return self.session.query(Watchlist).filter(
            Watchlist.id == watchlist_id,
            Watchlist.user_id == self.user_id
        ).first()
    
    def create_watchlist(self, name):
        """Create a new watchlist"""
        watchlist = Watchlist(name=name, user_id=self.user_id)
        self.session.add(watchlist)
        self.session.commit()
        return watchlist
    
    def add_to_watchlist(self, watchlist_id, symbol):
        """Add a stock to a watchlist"""
        # Check if item already exists in watchlist
        item = self.session.query(WatchlistItem).filter(
            WatchlistItem.watchlist_id == watchlist_id,
            WatchlistItem.symbol == symbol
        ).first()
        
        if not item:
            item = WatchlistItem(
                watchlist_id=watchlist_id,
                symbol=symbol
            )
            self.session.add(item)
            self.session.commit()
            
        return item
    
    def remove_from_watchlist(self, watchlist_id, symbol):
        """Remove a stock from a watchlist"""
        item = self.session.query(WatchlistItem).filter(
            WatchlistItem.watchlist_id == watchlist_id,
            WatchlistItem.symbol == symbol
        ).first()
        
        if item:
            self.session.delete(item)
            self.session.commit()
            return True
        
        return False
    
    def get_watchlist_items(self, watchlist_id):
        """Get all items in a specific watchlist"""
        return self.session.query(WatchlistItem).filter(WatchlistItem.watchlist_id == watchlist_id).all()
    
    def close(self):
        """Close the database session"""
        if self.session is not None:
            try:
                self.session.close()
            except Exception as e:
                print(f"Error closing database session: {e}")