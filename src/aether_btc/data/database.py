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


# ---------------------------------------------------------------------------
# Supported timeframes and their table suffixes
# ---------------------------------------------------------------------------
TIMEFRAMES = ("15m", "1h", "4h", "1d")

# Map interval string to Binance kline interval
BINANCE_INTERVALS = {
    "15m": "15m",
    "1h": "1h",
    "4h": "4h",
    "1d": "1d",
}

# Lookback bars for indicator warmup per timeframe
INDICATOR_LOOKBACK = {
    "15m": 500,
    "1h": 500,
    "4h": 500,
    "1d": 500,
}


def _candle_table_name(tf: str) -> str:
    return f"candles_{tf}"


def _indicators_table_name(tf: str) -> str:
    return f"btc_indicators_{tf}"


def _live_state_table_name(tf: str) -> str:
    return f"btc_live_state_{tf}"


# ---------------------------------------------------------------------------
# Candle models — one table per timeframe
# ---------------------------------------------------------------------------
def _make_candle_model(tf: str):
    table_name = _candle_table_name(tf)
    uq_name = f"uq_candle_{tf}_pair_ts"

    # Use type() to create unique class names that SQLAlchemy won't warn about
    attrs = {
        "__tablename__": table_name,
        "__table_args__": (UniqueConstraint("pair", "timestamp", name=uq_name),),
        "id": Column(Integer, primary_key=True, autoincrement=True),
        "pair": Column(String(20), nullable=False),
        "timestamp": Column(DateTime(timezone=True), nullable=False),
        "open": Column(Float, nullable=False),
        "high": Column(Float, nullable=False),
        "low": Column(Float, nullable=False),
        "close": Column(Float, nullable=False),
        "volume": Column(Float, nullable=False),
        "quote_volume": Column(Float),
        "trades_count": Column(Integer),
    }
    return type(f"Candle_{tf}", (Base,), attrs)


def _make_indicators_model(tf: str):
    table_name = _indicators_table_name(tf)
    uq_name = f"uq_indicators_{tf}_pair_ts"

    attrs = {
        "__tablename__": table_name,
        "__table_args__": (UniqueConstraint("pair", "timestamp", name=uq_name),),
        "id": Column(Integer, primary_key=True, autoincrement=True),
        "pair": Column(String(20), nullable=False),
        "timestamp": Column(DateTime(timezone=True), nullable=False),
        "open": Column(Float),
        "high": Column(Float),
        "low": Column(Float),
        "close": Column(Float),
        "volume": Column(Float),
        "indicators": Column(JSONB),
        "signal_type": Column(String(10)),
        "signal_score": Column(Float),
        "signal_confidence": Column(Float),
    }
    return type(f"BtcIndicators_{tf}", (Base,), attrs)


def _make_live_state_model(tf: str):
    table_name = _live_state_table_name(tf)

    attrs = {
        "__tablename__": table_name,
        "id": Column(Integer, primary_key=True, autoincrement=True),
        "pair": Column(String(20), nullable=False, unique=True),
        "timestamp": Column(DateTime(timezone=True), nullable=False),
        "price": Column(Float, nullable=False),
        "signal_type": Column(String(10)),
        "signal_score": Column(Float),
        "signal_confidence": Column(Float),
        "indicators": Column(JSONB),
        "strategy_signals": Column(JSONB),
        "model_name": Column(String(100)),
        "model_fitness": Column(Float),
        "updated_at": Column(DateTime(timezone=True), default=datetime.now, onupdate=datetime.now),
    }
    return type(f"BtcLiveState_{tf}", (Base,), attrs)


# Build model registries
CANDLE_MODELS: dict[str, type] = {}
INDICATOR_MODELS: dict[str, type] = {}
LIVE_STATE_MODELS: dict[str, type] = {}

for _tf in TIMEFRAMES:
    CANDLE_MODELS[_tf] = _make_candle_model(_tf)
    INDICATOR_MODELS[_tf] = _make_indicators_model(_tf)
    LIVE_STATE_MODELS[_tf] = _make_live_state_model(_tf)

# Convenience aliases for backward compatibility
Candle15m = CANDLE_MODELS["15m"]
BtcIndicators15m = INDICATOR_MODELS["15m"]
BtcLiveState = LIVE_STATE_MODELS["15m"]


# ---------------------------------------------------------------------------
# Non-timeframe tables (unchanged)
# ---------------------------------------------------------------------------
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


class WalkForwardProgress(Base):
    """Walk-forward window-level progress tracking."""

    __tablename__ = "walk_forward_progress"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String(50), nullable=False)
    total_windows = Column(Integer, nullable=False)
    current_window = Column(Integer, nullable=False)
    status = Column(String(20), nullable=False, default="training")
    train_start = Column(String(30))
    train_end = Column(String(30))
    test_start = Column(String(30))
    test_end = Column(String(30))
    train_fitness = Column(Float)
    oos_sharpe = Column(Float)
    oos_return = Column(Float)
    oos_max_dd = Column(Float)
    oos_trades = Column(Integer)
    oos_win_rate = Column(Float)
    oos_calmar = Column(Float)
    generations_run = Column(Integer)
    window_elapsed_seconds = Column(Float)
    created_at = Column(DateTime(timezone=True), default=datetime.now)
    updated_at = Column(DateTime(timezone=True), default=datetime.now, onupdate=datetime.now)

    __table_args__ = (
        UniqueConstraint("run_id", "current_window", name="uq_wf_run_window"),
    )


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


# ---------------------------------------------------------------------------
# Engine / Session helpers
# ---------------------------------------------------------------------------
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
