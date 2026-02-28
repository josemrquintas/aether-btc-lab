"""Data pipeline: orchestrate fetch -> store -> compute indicators."""

from datetime import datetime, timezone

import pandas as pd
import structlog
from sqlalchemy import text
from sqlalchemy.orm import Session

from aether_btc.data.binance_client import BinanceDataClient
from aether_btc.data.database import Candle15m, FundingRate, get_session

log = structlog.get_logger()


class DataPipeline:
    """Orchestrates data ingestion from Binance to PostgreSQL."""

    def __init__(self, session: Session | None = None, binance_client: BinanceDataClient | None = None) -> None:
        self.session = session or get_session()
        self.binance = binance_client or BinanceDataClient()

    def fetch_and_store_candles(
        self,
        pair: str = "BTCUSDT",
        interval: str = "15m",
        start: str = "1 Sep 2019",
        end: str | None = None,
    ) -> int:
        """Fetch candles from Binance and store in database. Returns count stored."""
        df = self.binance.fetch_klines(pair=pair, interval=interval, start=start, end=end)
        if df.empty:
            return 0

        stored = 0
        batch_size = 5000

        for i in range(0, len(df), batch_size):
            batch = df.iloc[i:i + batch_size]
            for _, row in batch.iterrows():
                candle = Candle15m(
                    pair=pair,
                    timestamp=row["timestamp"],
                    open=row["open"],
                    high=row["high"],
                    low=row["low"],
                    close=row["close"],
                    volume=row["volume"],
                    quote_volume=row["quote_volume"],
                    trades_count=row["trades_count"],
                )
                self.session.merge(candle)
                stored += 1

            self.session.commit()
            log.info("batch_stored", pair=pair, count=stored, total=len(df))

        log.info("candles_stored", pair=pair, total=stored)
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

        stored = 0
        for _, row in df.iterrows():
            rate = FundingRate(
                pair=pair,
                timestamp=row["timestamp"],
                funding_rate=row["funding_rate"],
            )
            self.session.merge(rate)
            stored += 1

        self.session.commit()
        log.info("funding_rates_stored", pair=pair, total=stored)
        return stored

    def load_candles(
        self,
        pair: str = "BTCUSDT",
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> pd.DataFrame:
        """Load candles from database as DataFrame."""
        query = self.session.query(Candle15m).filter(Candle15m.pair == pair)
        if start:
            query = query.filter(Candle15m.timestamp >= start)
        if end:
            query = query.filter(Candle15m.timestamp <= end)
        query = query.order_by(Candle15m.timestamp)

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

    def get_candle_count(self, pair: str = "BTCUSDT") -> int:
        """Get total candle count for a pair."""
        return self.session.query(Candle15m).filter(Candle15m.pair == pair).count()

    def get_funding_rate_count(self, pair: str = "BTCUSDT") -> int:
        """Get total funding rate count for a pair."""
        return self.session.query(FundingRate).filter(FundingRate.pair == pair).count()
