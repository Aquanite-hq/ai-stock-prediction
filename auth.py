import streamlit as st
from database import User, get_session
from datetime import datetime

def initialize_auth():
    """Initialize session state variables for authentication"""
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if 'user_id' not in st.session_state:
        st.session_state.user_id = None
    if 'username' not in st.session_state:
        st.session_state.username = None

def login_user(username, password):
    """
    Authenticate a user with username and password
    
    Args:
        username (str): Username
        password (str): Password
        
    Returns:
        bool: True if login successful, False otherwise
    """
    from sqlalchemy import text
    
    session = get_session()
    if session is None:
        print("Cannot login: no database session")
        st.error("Database connection error. Login is not available at this time.")
        return False
    
    try:
        # Use text SQL for better error handling
        user = session.query(User).filter(User.username == username).first()
        
        if user and user.check_password(password):
            # Update last login time
            user.last_login = datetime.now()
            session.commit()
            
            # Set session state
            st.session_state.logged_in = True
            st.session_state.user_id = user.id
            st.session_state.username = user.username
            
            return True
        
        return False
    except Exception as e:
        print(f"Login error: {e}")
        st.error(f"Login error: {str(e)}")
        try:
            session.rollback()
        except Exception as rollback_err:
            print(f"Error rolling back transaction: {rollback_err}")
        return False
    finally:
        if session is not None:
            try:
                session.close()
            except Exception as e:
                print(f"Error closing session during login: {e}")

def register_user(username, email, password):
    """
    Register a new user
    
    Args:
        username (str): Username
        email (str): Email address
        password (str): Password
        
    Returns:
        tuple: (bool, str) - (success, message)
    """
    session = get_session()
    if session is None:
        print("Cannot register: no database session")
        return False, "Database connection error. Registration is not available at this time."
    
    try:
        # Check if username already exists
        existing_user = session.query(User).filter(User.username == username).first()
        if existing_user:
            return False, "Username already exists"
        
        # Check if email already exists
        if email:
            existing_email = session.query(User).filter(User.email == email).first()
            if existing_email:
                return False, "Email already exists"
        
        # Create password hash
        password_hash, salt = User.hash_password(password)
        
        # Create new user
        new_user = User(
            username=username,
            email=email,
            password_hash=password_hash,
            salt=salt,
            created_at=datetime.now()
        )
        
        session.add(new_user)
        
        # Create default portfolio and watchlist for the new user
        from database import Portfolio, Watchlist
        
        default_portfolio = Portfolio(name="My Portfolio", user=new_user)
        session.add(default_portfolio)
        
        default_watchlist = Watchlist(name="My Watchlist", user=new_user)
        session.add(default_watchlist)
        
        session.commit()
        
        return True, "Registration successful"
    except Exception as e:
        print(f"Registration error: {e}")
        try:
            session.rollback()
        except Exception as rollback_err:
            print(f"Error rolling back transaction: {rollback_err}")
        return False, f"Registration error: {str(e)}"
    finally:
        if session is not None:
            try:
                session.close()
            except Exception as e:
                print(f"Error closing session during registration: {e}")

def logout_user():
    """Log out the current user"""
    st.session_state.logged_in = False
    st.session_state.user_id = None
    st.session_state.username = None

def is_logged_in():
    """Check if a user is logged in"""
    return st.session_state.logged_in

def get_current_user_id():
    """Get the current user ID"""
    return st.session_state.user_id

def show_login_page():
    """Display the login form"""
    st.title("Login")
    
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        
        col1, col2 = st.columns(2)
        with col1:
            login_button = st.form_submit_button("Login")
        with col2:
            register_toggle = st.checkbox("New user? Register here")
    
    if login_button and not register_toggle:
        if username and password:
            if login_user(username, password):
                st.success("Login successful!")
                # Add a rerun to update the UI
                st.rerun()
            else:
                st.error("Invalid username or password")
        else:
            st.warning("Please enter both username and password")
    
    if register_toggle:
        show_registration_form()

def show_registration_form():
    """Display the registration form"""
    st.subheader("Register")
    
    with st.form("register_form"):
        username = st.text_input("Username")
        email = st.text_input("Email (optional)")
        password = st.text_input("Password", type="password")
        confirm_password = st.text_input("Confirm Password", type="password")
        
        register_button = st.form_submit_button("Register")
    
    if register_button:
        if not username or not password:
            st.warning("Username and password are required")
        elif password != confirm_password:
            st.error("Passwords do not match")
        else:
            success, message = register_user(username, email, password)
            if success:
                st.success(message)
                # Log in the new user
                login_user(username, password)
                st.rerun()
            else:
                st.error(message)