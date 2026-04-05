"""Data pipeline: orchestrate fetch -> store -> compute indicators."""

from datetime import datetime, timezone

import pandas as pd
import structlog
from sqlalchemy import text
from sqlalchemy.orm import Session

from aether_btc.data.binance_client import BinanceDataClient
from aether_btc.data.database import (
    CANDLE_MODELS,
    FundingRate,
    _candle_table_name,
    get_engine,
    get_session,
)

log = structlog.get_logger()


class DataPipeline:
    """Orchestrates data ingestion from Binance to PostgreSQL."""

    def __init__(self, session: Session | None = None, binance_client: BinanceDataClient | None = None) -> None:
        self.session = session or get_session()
        self.engine = self.session.bind
        self.binance = binance_client or BinanceDataClient()

    def fetch_and_store_candles(
        self,
        pair: str = "BTCUSDT",
        interval: str = "15m",
        start: str = "1 Sep 2019",
        end: str | None = None,
    ) -> int:
        """Fetch candles from Binance and store in database. Returns count stored."""
        table = _candle_table_name(interval)
        df = self.binance.fetch_klines(pair=pair, interval=interval, start=start, end=end)
        if df.empty:
            return 0

        df["pair"] = pair
        batch_size = 10000
        stored = 0

        for i in range(0, len(df), batch_size):
            batch = df.iloc[i:i + batch_size]
            records = batch[["pair", "timestamp", "open", "high", "low", "close",
                             "volume", "quote_volume", "trades_count"]].to_dict("records")

            with self.engine.connect() as conn:
                conn.execute(
                    text(f"""
                        INSERT INTO {table} (pair, timestamp, open, high, low, close, volume, quote_volume, trades_count)
                        VALUES (:pair, :timestamp, :open, :high, :low, :close, :volume, :quote_volume, :trades_count)
                        ON CONFLICT (pair, timestamp) DO NOTHING
                    """),
                    records,
                )
                conn.commit()

            stored += len(records)
            log.info("batch_stored", pair=pair, interval=interval, count=stored, total=len(df))

        log.info("candles_stored", pair=pair, interval=interval, total=stored)
        return stored

    def fetch_and_store_funding_rates(
        self,
        pair: str = "BTCUSDT",
        start: str = "1 Sep 2019",
        end: str | None = None,
    ) -> int:
        """Fetch funding rates from Binance and store in database."""
        df = self.binance.fetch_funding_rates(pair=pair, start=start, end=end)
        if df.empty:
            return 0

        df["pair"] = pair
        records = df[["pair", "timestamp", "funding_rate"]].to_dict("records")

        with self.engine.connect() as conn:
            conn.execute(
                text("""
                    INSERT INTO funding_rates (pair, timestamp, funding_rate)
                    VALUES (:pair, :timestamp, :funding_rate)
                    ON CONFLICT (pair, timestamp) DO NOTHING
                """),
                records,
            )
            conn.commit()

        log.info("funding_rates_stored", pair=pair, total=len(records))
        return len(records)

    def load_candles(
        self,
        pair: str = "BTCUSDT",
        interval: str = "15m",
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> pd.DataFrame:
        """Load candles from database as DataFrame."""
        model = CANDLE_MODELS[interval]
        query = self.session.query(model).filter(model.pair == pair)
        if start:
            query = query.filter(model.timestamp >= start)
        if end:
            query = query.filter(model.timestamp <= end)
        query = query.order_by(model.timestamp)

        rows = query.all()
        if not rows:
            return pd.DataFrame()

        data = [{
            "timestamp": r.timestamp,
            "open": r.open,
            "high": r.high,
            "low": r.low,
            "close": r.close,
            "volume": r.volume,
            "quote_volume": r.quote_volume,
            "trades_count": r.trades_count,
        } for r in rows]

        df = pd.DataFrame(data)
        df.set_index("timestamp", inplace=True)
        return df

    def load_funding_rates(
        self,
        pair: str = "BTCUSDT",
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> pd.DataFrame:
        """Load funding rates from database as DataFrame."""
        query = self.session.query(FundingRate).filter(FundingRate.pair == pair)
        if start:
            query = query.filter(FundingRate.timestamp >= start)
        if end:
            query = query.filter(FundingRate.timestamp <= end)
        query = query.order_by(FundingRate.timestamp)

        rows = query.all()
        if not rows:
            return pd.DataFrame()

        data = [{"timestamp": r.timestamp, "funding_rate": r.funding_rate} for r in rows]
        df = pd.DataFrame(data)
        df.set_index("timestamp", inplace=True)
        return df

    def get_candle_count(self, pair: str = "BTCUSDT", interval: str = "15m") -> int:
        """Get total candle count for a pair."""
        model = CANDLE_MODELS[interval]
        return self.session.query(model).filter(model.pair == pair).count()

    def get_funding_rate_count(self, pair: str = "BTCUSDT") -> int:
        """Get total funding rate count for a pair."""
        return self.session.query(FundingRate).filter(FundingRate.pair == pair).count()
