# DB config and initialization
# SQLAlchemy database operations
# -*- coding: utf-8 -*-
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.engine.url import URL



DATABASE = {
    'drivername': 'postgresql',
    'host': 'localhost',  
    'port': '5432',      
    'username': 'postgres',
    'password': 'postgres',
    'database': 'Inventory'
}

# Create the connection URL with proper encoding
url = URL.create(**DATABASE)
engine = create_engine(
    url,
    client_encoding='utf8',
    connect_args={'client_encoding': 'utf8'}
)

#Session Configuration
session_factory = sessionmaker(bind=engine) #Session factory creates new Session objects when called. The bind parameter specifies the engine to use for the session.
Session = scoped_session(session_factory) #Scoped session allows for thread-local sessions, which is useful in web applications where each request can have its own session


# Provides a context-managed database session.
def get_db_session():
    session = Session()
    try :
        yield session # This will yield the session to the caller, allowing them to use it within a context manager.
    finally:
        session.close()  # Close the session when done. This ensures that the session is properly cleaned up after use.



