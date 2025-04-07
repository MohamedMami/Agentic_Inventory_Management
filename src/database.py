# DB config and initialization
# SQLAlchemy database operations

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, scoped_session

from config import DATABASE_URL


# Create a new SQLAlchemy engine instance
engine = create_engine(DATABASE_URL, echo=False)# echo = false Disables logging of raw SQL statements (useful in production to avoid verbose logs)

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
    


