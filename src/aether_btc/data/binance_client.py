"""Binance API client for fetching historical klines and funding rates.

Uses public endpoints for historical data (no Futures perms needed).
Rate limiting and pagination handled automatically.
"""

from datetime import datetime, timezone

import pandas as pd
import structlog
from binance.client import Client
from tqdm import tqdm

log = structlog.get_logger()

# Binance returns max 1000 candles per request
MAX_KLINES_PER_REQUEST = 1000
# Funding rates: max 1000 per request
MAX_FUNDING_PER_REQUEST = 1000


class BinanceDataClient:
    """Wrapper around python-binance for historical data fetching."""

    def __init__(self, api_key: str = "", api_secret: str = "") -> None:
        self.client = Client(api_key or "", api_secret or "")

    def fetch_klines(
        self,
        pair: str = "BTCUSDT",
        interval: str = "15m",
        start: str = "1 Sep 2019",
        end: str | None = None,
    ) -> pd.DataFrame:
        """Fetch historical klines with automatic pagination.

        Returns DataFrame with columns:
        timestamp, open, high, low, close, volume, quote_volume, trades_count
        """
        log.info("fetching_klines", pair=pair, interval=interval, start=start, end=end)

        klines = self.client.get_historical_klines(
            symbol=pair,
            interval=interval,
            start_str=start,
            end_str=end,
        )

        if not klines:
            log.warning("no_klines_returned", pair=pair)
            return pd.DataFrame()

        df = pd.DataFrame(klines, columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades_count",
            "taker_buy_volume", "taker_buy_quote_volume", "ignore",
        ])

        df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        df = df[["timestamp", "open", "high", "low", "close", "volume",
                  "quote_volume", "trades_count"]].copy()

        for col in ["open", "high", "low", "close", "volume", "quote_volume"]:
            df[col] = df[col].astype(float)
        df["trades_count"] = df["trades_count"].astype(int)

        log.info("klines_fetched", pair=pair, count=len(df))
        return df

    def fetch_funding_rates(
        self,
        pair: str = "BTCUSDT",
        start: str = "1 Sep 2019",
        end: str | None = None,
    ) -> pd.DataFrame:
        """Fetch historical funding rates with pagination.

        Binance Futures funding rates: every 8h (00:00, 08:00, 16:00 UTC).
        Returns DataFrame with columns: timestamp, funding_rate
        """
        log.info("fetching_funding_rates", pair=pair, start=start)

        start_ts = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
        end_ts = int(pd.Timestamp(end, tz="UTC").timestamp() * 1000) if end else None

        all_rates: list[dict] = []
        current_start = start_ts

        while True:
            params: dict = {
                "symbol": pair,
                "startTime": current_start,
                "limit": MAX_FUNDING_PER_REQUEST,
            }
            if end_ts:
                params["endTime"] = end_ts

            rates = self.client.futures_funding_rate(**params)

            if not rates:
                break

            all_rates.extend(rates)

            if len(rates) < MAX_FUNDING_PER_REQUEST:
                break

            # Move past last returned timestamp
            current_start = rates[-1]["fundingTime"] + 1

        if not all_rates:
            log.warning("no_funding_rates_returned", pair=pair)
            return pd.DataFrame()

        df = pd.DataFrame(all_rates)
        df["timestamp"] = pd.to_datetime(df["fundingTime"], unit="ms", utc=True)
        df["funding_rate"] = df["fundingRate"].astype(float)
        df = df[["timestamp", "funding_rate"]].copy()

        # Remove duplicates
        df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

        log.info("funding_rates_fetched", pair=pair, count=len(df))
        return df
