import os
import sys
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, ForeignKey, Boolean, MetaData, Table, inspect, text
from datetime import datetime

# Get database URL from environment variable
DATABASE_URL = os.environ.get('DATABASE_URL')

# Create engine
engine = create_engine(DATABASE_URL)

# Create a metadata object
metadata = MetaData()

# Connect to the database and get a connection
conn = engine.connect()

# Start a transaction
trans = conn.begin()

try:
    # Check if table exists
    inspector = inspect(engine)
    tables = inspector.get_table_names()
    
    if 'users' in tables:
        # Get columns in users table
        columns = [column['name'] for column in inspector.get_columns('users')]
        
        # Check if password_hash column exists
        if 'password_hash' not in columns:
            print("Adding password_hash column to users table...")
            conn.execute(text('ALTER TABLE users ADD COLUMN password_hash VARCHAR DEFAULT NULL'))
        
        # Check if salt column exists
        if 'salt' not in columns:
            print("Adding salt column to users table...")
            conn.execute(text('ALTER TABLE users ADD COLUMN salt VARCHAR DEFAULT NULL'))
        
        # Check if last_login column exists
        if 'last_login' not in columns:
            print("Adding last_login column to users table...")
            conn.execute(text('ALTER TABLE users ADD COLUMN last_login TIMESTAMP DEFAULT NULL'))
        
        # Import User class after we've made schema modifications to avoid import errors
        print("Updating existing users with default password...")
        # Import hashlib and uuid directly to create password hash and salt
        import hashlib
        import uuid
        
        # Generate salt and password hash
        salt = uuid.uuid4().hex
        password = "password"
        hashed_password = hashlib.sha256((password + salt).encode()).hexdigest()
        
        # Update existing users with default password
        conn.execute(text(f"UPDATE users SET password_hash = '{hashed_password}', salt = '{salt}' WHERE password_hash IS NULL"))
        
        print("Database migration completed successfully!")
    else:
        print("Users table does not exist. The tables will be created with the new schema.")
    
    # Commit the transaction
    trans.commit()
    
except Exception as e:
    # If there is an error, rollback the transaction
    trans.rollback()
    print(f"Error during migration: {e}")
    sys.exit(1)

finally:
    # Close the connection
    conn.close()