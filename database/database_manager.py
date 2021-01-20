"""Some utils for database.
"""
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from .base import Base

session = None

def get_session(url_path) -> Session:
    """Creates engine and database if needed and returns session

    Arguments:
        url_path {str} -- Path to sqlite file

    Returns:
        Session -- Session for the database
    """
    global session
    if not session:
        engine = create_engine(f"sqlite:///{url_path}")
        Base.metadata.bind = engine
        Base.metadata.create_all()
        Session_maked = sessionmaker(bind=engine)
        session = Session_maked()

    return session
