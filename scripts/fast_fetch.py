#!/usr/bin/env python3
"""Fast bulk fetch using PostgreSQL COPY protocol — 100x faster than INSERT."""
import io
import time

import psycopg2
from dotenv import load_dotenv

load_dotenv()

from aether_btc.data.binance_client import BinanceDataClient
from aether_btc.data.database import init_db
from aether_btc.core.config import DatabaseConfig

init_db()
cfg = DatabaseConfig()
client = BinanceDataClient()

# ---- Fetch from Binance ----
print("Fetching candles from Binance...")
t0 = time.time()
df = client.fetch_klines(pair="BTCUSDT", interval="15m", start="1 Sep 2019", end="28 Feb 2026")
print(f"Fetched {len(df):,} bars in {time.time()-t0:.0f}s")

# ---- Bulk load via COPY ----
conn = psycopg2.connect(cfg.url)
cur = conn.cursor()

# Clear old data
print("Clearing old candles...")
cur.execute("DELETE FROM candles_15m WHERE pair = 'BTCUSDT'")
conn.commit()

# Build CSV in memory
print("Building CSV buffer...")
t1 = time.time()
buf = io.StringIO()
for _, row in df.iterrows():
    buf.write(f"BTCUSDT\t{row['timestamp']}\t{row['open']}\t{row['high']}\t{row['low']}\t{row['close']}\t{row['volume']}\t{row['quote_volume']}\t{row['trades_count']}\n")
buf.seek(0)
print(f"CSV built in {time.time()-t1:.0f}s")

# COPY FROM stdin
print("COPY loading into DB...")
t2 = time.time()
cur.copy_from(buf, "candles_15m", columns=("pair", "timestamp", "open", "high", "low", "close", "volume", "quote_volume", "trades_count"))
conn.commit()
print(f"COPY done in {time.time()-t2:.0f}s")

# ---- Funding rates ----
print("\nFetching funding rates...")
t3 = time.time()
fdf = client.fetch_funding_rates(pair="BTCUSDT", start="1 Sep 2019", end="28 Feb 2026")
print(f"Fetched {len(fdf):,} funding rates in {time.time()-t3:.0f}s")

if not fdf.empty:
    cur.execute("DELETE FROM funding_rates WHERE pair = 'BTCUSDT'")
    conn.commit()
    fbuf = io.StringIO()
    for _, row in fdf.iterrows():
        fbuf.write(f"BTCUSDT\t{row['timestamp']}\t{row['funding_rate']}\n")
    fbuf.seek(0)
    cur.copy_from(fbuf, "funding_rates", columns=("pair", "timestamp", "funding_rate"))
    conn.commit()
    print(f"Stored {len(fdf):,} funding rates")

# ---- Verify ----
cur.execute("SELECT COUNT(*), MIN(timestamp), MAX(timestamp) FROM candles_15m WHERE pair='BTCUSDT'")
r = cur.fetchone()
cur.execute("SELECT COUNT(*) FROM funding_rates WHERE pair='BTCUSDT'")
r2 = cur.fetchone()
cur.close()
conn.close()

print(f"\nDB candles: {r[0]:,} bars from {r[1]} to {r[2]}")
print(f"DB funding: {r2[0]:,} rates")
days = (r[2] - r[1]).days if r[1] and r[2] else 0
print(f"Data span: {days} days (~{days // 30} months)")
print(f"Expected walk-forward windows (180d train + 60d test): ~{max(0, (days - 180) // 60)}")
print(f"\nTotal time: {time.time()-t0:.0f}s")
