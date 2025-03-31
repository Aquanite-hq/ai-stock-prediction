import os
import time
import streamlit as st
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, ForeignKey, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.exc import OperationalError, SQLAlchemyError
from datetime import datetime
import hashlib
import uuid

# Get database URL from environment variable
DATABASE_URL = os.environ.get('DATABASE_URL')

# Create engine with connection pool settings and retry logic
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    pool_recycle=3600,
    connect_args={"connect_timeout": 30}
)

# Create base class
Base = declarative_base()

# Define User model
class User(Base):
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True)
    username = Column(String, unique=True, nullable=False)
    email = Column(String, unique=True)
    password_hash = Column(String, nullable=False)
    salt = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.now)
    last_login = Column(DateTime, nullable=True)
    
    # Define relationships
    portfolios = relationship("Portfolio", back_populates="user", cascade="all, delete-orphan")
    watchlists = relationship("Watchlist", back_populates="user", cascade="all, delete-orphan")
    
    @staticmethod
    def hash_password(password, salt=None):
        """Hash a password with a salt or generate a new salt"""
        if salt is None:
            salt = uuid.uuid4().hex
        
        # Combine password and salt, then hash
        hashed_password = hashlib.sha256((password + salt).encode()).hexdigest()
        return hashed_password, salt
    
    def check_password(self, password):
        """Check if provided password matches the stored hash"""
        hashed_input, _ = self.hash_password(password, self.salt)
        return hashed_input == self.password_hash

# Define Portfolio model
class Portfolio(Base):
    __tablename__ = 'portfolios'
    
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    user_id = Column(Integer, ForeignKey('users.id'))
    created_at = Column(DateTime, default=datetime.now)
    
    # Define relationships
    user = relationship("User", back_populates="portfolios")
    holdings = relationship("Holding", back_populates="portfolio", cascade="all, delete-orphan")
    transactions = relationship("Transaction", back_populates="portfolio", cascade="all, delete-orphan")

# Define Holding model
class Holding(Base):
    __tablename__ = 'holdings'
    
    id = Column(Integer, primary_key=True)
    portfolio_id = Column(Integer, ForeignKey('portfolios.id'))
    symbol = Column(String, nullable=False)
    quantity = Column(Float, nullable=False)
    average_price = Column(Float, nullable=False)
    
    # Define relationships
    portfolio = relationship("Portfolio", back_populates="holdings")

# Define Transaction model
class Transaction(Base):
    __tablename__ = 'transactions'
    
    id = Column(Integer, primary_key=True)
    portfolio_id = Column(Integer, ForeignKey('portfolios.id'))
    symbol = Column(String, nullable=False)
    action = Column(String, nullable=False)  # 'buy' or 'sell'
    quantity = Column(Float, nullable=False)
    price = Column(Float, nullable=False)
    timestamp = Column(DateTime, default=datetime.now)
    
    # Define relationships
    portfolio = relationship("Portfolio", back_populates="transactions")

# Define Watchlist model
class Watchlist(Base):
    __tablename__ = 'watchlists'
    
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    user_id = Column(Integer, ForeignKey('users.id'))
    created_at = Column(DateTime, default=datetime.now)
    
    # Define relationships
    user = relationship("User", back_populates="watchlists")
    items = relationship("WatchlistItem", back_populates="watchlist", cascade="all, delete-orphan")

# Define WatchlistItem model
class WatchlistItem(Base):
    __tablename__ = 'watchlist_items'
    
    id = Column(Integer, primary_key=True)
    watchlist_id = Column(Integer, ForeignKey('watchlists.id'))
    symbol = Column(String, nullable=False)
    added_at = Column(DateTime, default=datetime.now)
    
    # Define relationships
    watchlist = relationship("Watchlist", back_populates="items")

# Initialize database if it doesn't exist
def init_db():
    max_retries = 5
    retry_count = 0
    retry_delay = 2  # seconds
    
    while retry_count < max_retries:
        try:
            Base.metadata.create_all(engine)
            print("Database initialized successfully")
            return True
        except (OperationalError, SQLAlchemyError) as e:
            retry_count += 1
            if retry_count >= max_retries:
                st.error(f"Failed to connect to database after {max_retries} attempts: {e}")
                print(f"Failed to connect to database after {max_retries} attempts: {e}")
                return False
            print(f"Database connection error (attempt {retry_count}/{max_retries}): {e}")
            print(f"Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)
            retry_delay *= 2  # Exponential backoff

# Function to test if database connection is working
def test_database_connection():
    """Test if database connection is working"""
    from sqlalchemy import text
    
    max_retries = 3
    retry_count = 0
    retry_delay = 1  # seconds
    
    while retry_count < max_retries:
        try:
            # Create a test connection and run a simple query
            test_session = sessionmaker(bind=engine)()
            # Run a simple query that doesn't require tables to exist
            test_session.execute(text("SELECT 1"))
            test_session.close()
            print("Database connection test successful")
            return True
        except (OperationalError, SQLAlchemyError) as e:
            retry_count += 1
            if retry_count >= max_retries:
                print(f"Database connection test failed after {max_retries} attempts: {e}")
                return False
            print(f"Database connection test error (attempt {retry_count}/{max_retries}): {e}")
            time.sleep(retry_delay)
            retry_delay *= 2  # Exponential backoff

# Create a session with retry logic
def get_session():
    max_retries = 3
    retry_count = 0
    retry_delay = 1  # seconds
    
    # First test if the database is accessible
    if not test_database_connection():
        st.error("Cannot connect to database. Please check your database configuration.")
        print("Database connection test failed before creating session.")
        return None
    
    while retry_count < max_retries:
        try:
            Session = sessionmaker(bind=engine)
            return Session()
        except (OperationalError, SQLAlchemyError) as e:
            retry_count += 1
            if retry_count >= max_retries:
                st.error(f"Failed to create database session after {max_retries} attempts: {e}")
                raise
            print(f"Session creation error (attempt {retry_count}/{max_retries}): {e}")
            time.sleep(retry_delay)
            retry_delay *= 2  # Exponential backoff

# Try to initialize the database when module is imported
try:
    init_db()
except Exception as e:
    print(f"Error during initial database initialization: {e}")

# Create a default user if there are no users in the database
def get_or_create_default_user():
    from sqlalchemy import text
    
    session = get_session()
    if session is None:
        print("Could not create session for get_or_create_default_user")
        st.error("Database connection error. Cannot access user information.")
        return None
    
    try:
        # Check if there are any users - use text() to create the query
        try:
            result = session.execute(text("SELECT COUNT(*) FROM users"))
            user_count = result.scalar()
            
            if user_count == 0:
                # Create a default user with password "password"
                password_hash, salt = User.hash_password("password")
                default_user = User(
                    username="default_user", 
                    email="default@example.com",
                    password_hash=password_hash,
                    salt=salt
                )
                session.add(default_user)
                
                # Create a default portfolio
                default_portfolio = Portfolio(name="My Portfolio", user=default_user)
                session.add(default_portfolio)
                
                # Create a default watchlist
                default_watchlist = Watchlist(name="My Watchlist", user=default_user)
                session.add(default_watchlist)
                
                try:
                    session.commit()
                    print("Default user created successfully")
                    return default_user
                except (OperationalError, SQLAlchemyError) as e:
                    session.rollback()
                    print(f"Error committing default user: {e}")
                    st.error(f"Database error: {e}")
                    return None
            else:
                # Return the first user
                first_user = session.execute(text("SELECT * FROM users LIMIT 1")).fetchone()
                if first_user:
                    # Create a User object from the row data
                    return User(
                        id=first_user.id, 
                        username=first_user.username,
                        email=first_user.email,
                        password_hash=first_user.password_hash,
                        salt=first_user.salt
                    )
                return None
        except (OperationalError, SQLAlchemyError) as e:
            print(f"Error querying users: {e}")
            st.error(f"Database error: {e}")
            return None
    except Exception as e:
        print(f"Error in get_or_create_default_user: {e}")
        st.error(f"Database error: {e}")
        return None
    finally:
        # Make sure we always close the session if it exists
        if session is not None:
            try:
                session.close()
                print("Session closed successfully")
            except Exception as e:
                print(f"Error closing session: {e}")

# Initialize the function to get or create a default user
def init_app():
    max_retries = 3
    retry_count = 0
    retry_delay = 2  # seconds
    
    while retry_count < max_retries:
        try:
            # Create a default user if there are no users
            user = get_or_create_default_user()
            if user:
                return user
            
            # If user is None, something went wrong, retry
            retry_count += 1
            if retry_count >= max_retries:
                st.error("Failed to initialize application. Please restart.")
                print("Failed to initialize application after multiple attempts.")
                # Return a dummy user object to avoid errors
                return type('obj', (object,), {'id': 1, 'username': 'system'})
            
            print(f"Retrying app initialization (attempt {retry_count}/{max_retries})...")
            time.sleep(retry_delay)
            retry_delay *= 2  # Exponential backoff
        except Exception as e:
            retry_count += 1
            print(f"Error in app initialization (attempt {retry_count}/{max_retries}): {e}")
            if retry_count >= max_retries:
                st.error(f"Failed to initialize application: {e}")
                # Return a dummy user object to avoid errors
                return type('obj', (object,), {'id': 1, 'username': 'system'})
            time.sleep(retry_delay)
            retry_delay *= 2  # Exponential backoff