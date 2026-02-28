"""SQLAlchemy database models for Aether BTC."""

from datetime import datetime

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Integer,
    String,
    UniqueConstraint,
    create_engine,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from aether_btc.core.config import DatabaseConfig


class Base(DeclarativeBase):
    pass


class Candle15m(Base):
    """15-minute OHLCV candle data."""

    __tablename__ = "candles_15m"

    id = Column(Integer, primary_key=True, autoincrement=True)
    pair = Column(String(20), nullable=False)
    timestamp = Column(DateTime(timezone=True), nullable=False)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)
    quote_volume = Column(Float)
    trades_count = Column(Integer)

    __table_args__ = (UniqueConstraint("pair", "timestamp", name="uq_candle_pair_ts"),)


class FundingRate(Base):
    """Historical funding rates."""

    __tablename__ = "funding_rates"

    id = Column(Integer, primary_key=True, autoincrement=True)
    pair = Column(String(20), nullable=False)
    timestamp = Column(DateTime(timezone=True), nullable=False)
    funding_rate = Column(Float, nullable=False)

    __table_args__ = (UniqueConstraint("pair", "timestamp", name="uq_funding_pair_ts"),)


class GATrainingProgress(Base):
    """GA training progress tracking."""

    __tablename__ = "ga_training_progress"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String(50), nullable=False)
    generation = Column(Integer, nullable=False)
    best_fitness = Column(Float)
    avg_fitness = Column(Float)
    best_sharpe = Column(Float)
    best_calmar = Column(Float)
    best_max_drawdown = Column(Float)
    total_trades = Column(Integer)
    elapsed_seconds = Column(Float)
    created_at = Column(DateTime(timezone=True), default=datetime.now)

    __table_args__ = (UniqueConstraint("run_id", "generation", name="uq_ga_run_gen"),)


class Model(Base):
    """Trained GA models / chromosomes."""

    __tablename__ = "models"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(100))
    chromosome = Column(JSONB, nullable=False)
    fitness = Column(Float)
    sharpe = Column(Float)
    calmar = Column(Float)
    train_start = Column(DateTime)
    train_end = Column(DateTime)
    test_start = Column(DateTime)
    test_end = Column(DateTime)
    config = Column(JSONB)
    created_at = Column(DateTime(timezone=True), default=datetime.now)


class TradeLog(Base):
    """Live trades log."""

    __tablename__ = "trades"

    id = Column(Integer, primary_key=True, autoincrement=True)
    pair = Column(String(20))
    side = Column(String(10))
    leverage = Column(Float)
    entry_price = Column(Float)
    exit_price = Column(Float)
    quantity = Column(Float)
    margin_used = Column(Float)
    pnl = Column(Float)
    funding_cost = Column(Float)
    commission = Column(Float)
    entry_time = Column(DateTime(timezone=True))
    exit_time = Column(DateTime(timezone=True))
    exit_reason = Column(String(30))
    model_id = Column(Integer)
    created_at = Column(DateTime(timezone=True), default=datetime.now)


def get_engine(config: DatabaseConfig | None = None):
    """Create SQLAlchemy engine."""
    cfg = config or DatabaseConfig()
    return create_engine(cfg.url, pool_pre_ping=True)


def get_session(config: DatabaseConfig | None = None) -> Session:
    """Create a new database session."""
    engine = get_engine(config)
    SessionLocal = sessionmaker(bind=engine)
    return SessionLocal()


def init_db(config: DatabaseConfig | None = None) -> None:
    """Create all tables."""
    engine = get_engine(config)
    Base.metadata.create_all(engine)


def create_database_if_not_exists(config: DatabaseConfig | None = None) -> None:
    """Create the database if it doesn't exist."""
    cfg = config or DatabaseConfig()
    admin_url = f"postgresql://{cfg.user}:{cfg.password}@{cfg.host}:{cfg.port}/postgres"
    engine = create_engine(admin_url, isolation_level="AUTOCOMMIT")
    with engine.connect() as conn:
        result = conn.execute(
            text("SELECT 1 FROM pg_database WHERE datname = :name"),
            {"name": cfg.name},
        )
        if not result.fetchone():
            conn.execute(text(f'CREATE DATABASE "{cfg.name}"'))
