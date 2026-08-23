"""
db.py
SQLAlchemy engine and session setup for the shipping-delay-ai app.
Reads the database connection string from .env (DATABASE_URL).
"""

import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise RuntimeError(
        "DATABASE_URL is not set in .env. "
        "Expected format: mysql+pymysql://user:password@localhost:3306/shipping_db"
    )

engine = create_engine(DATABASE_URL, echo=False, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()


def get_db():
    """Yields a database session, auto-closing it after use (Flask dependency-style)."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Creates all tables defined in database/models.py if they don't already exist."""
    import database.models  # noqa: ensures models are registered on Base before create_all
    Base.metadata.create_all(bind=engine)