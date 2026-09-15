"""
ingestion.py
------------
Automated Data Ingestion layer for the Real-Time Predictive Dashboard capstone.

Responsibilities:
  * Pull live + recent-historical air quality readings from the free,
    key-less Open-Meteo Air Quality API.
  * Persist every reading into a local DuckDB database file so the
    Streamlit dashboard can query it with SQL.
  * Run either as a one-shot cycle (called from the Streamlit app's
    background thread) or as a standalone long-running poller
    (`python ingestion.py`).

This module is import-safe: importing it never starts network calls or
loops on its own -- callers explicitly invoke run_ingestion_cycle(),
backfill_historical(), or run_ingestion_loop().
"""

import time
import logging
import threading
from datetime import datetime, timezone
from typing import Optional

import requests
import duckdb
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S UTC",
)
log = logging.getLogger("ingestion")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
LATITUDE = -1.2921    # Nairobi, Kenya (default location; override in app UI)
LONGITUDE = 36.8219

BASE_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"

POLLUTANT_FIELDS = [
    "pm10",
    "pm2_5",
    "carbon_monoxide",
    "nitrogen_dioxide",
    "sulphur_dioxide",
    "ozone",
    "us_aqi",
    "european_aqi",
]

DB_PATH = "air_quality.duckdb"
TABLE_NAME = "air_quality_readings"
POLL_INTERVAL_SECONDS = 900  # 15 min -- Open-Meteo's AQ model refreshes hourly

# DuckDB only allows a single writer per on-disk file at a time. Since the
# Streamlit app and the ingestion loop share one process, we serialize all
# reads/writes through this lock to avoid "could not set lock" errors.
DB_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
def init_db(db_path: str = DB_PATH) -> None:
    """Create the readings table if it doesn't already exist."""
    with DB_LOCK:
        con = duckdb.connect(db_path)
        try:
            con.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
                    utc_timestamp     TIMESTAMP PRIMARY KEY,
                    latitude          DOUBLE,
                    longitude         DOUBLE,
                    pm10              DOUBLE,
                    pm2_5             DOUBLE,
                    carbon_monoxide   DOUBLE,
                    nitrogen_dioxide  DOUBLE,
                    sulphur_dioxide   DOUBLE,
                    ozone             DOUBLE,
                    us_aqi            DOUBLE,
                    european_aqi      DOUBLE
                )
                """
            )
        finally:
            con.close()


# ---------------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------------
def fetch_current_payload(
    latitude: float = LATITUDE, longitude: float = LONGITUDE
) -> Optional[dict]:
    """Fetch a single 'right now' reading. Fault-tolerant: returns None on error."""
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "current": ",".join(POLLUTANT_FIELDS),
        "timezone": "auto",
    }
    try:
        response = requests.get(BASE_URL, params=params, timeout=10)
        response.raise_for_status()
        payload = response.json()
        if "current" not in payload:
            log.error("API response missing 'current' block: %s", payload)
            return None
        return payload
    except requests.exceptions.RequestException as e:
        log.error("Network error during API fetch: %s", e)
        return None
    except ValueError as e:
        log.error("Invalid JSON response from API: %s", e)
        return None


def fetch_historical_payload(
    latitude: float = LATITUDE,
    longitude: float = LONGITUDE,
    past_days: int = 3,
) -> Optional[dict]:
    """Fetch hourly history for the last `past_days` days (used to seed the
    database on first run so charts aren't empty)."""
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": ",".join(POLLUTANT_FIELDS),
        "past_days": past_days,
        "forecast_days": 0,
        "timezone": "auto",
    }
    try:
        response = requests.get(BASE_URL, params=params, timeout=15)
        response.raise_for_status()
        payload = response.json()
        if "hourly" not in payload:
            log.error("API response missing 'hourly' block: %s", payload)
            return None
        return payload
    except requests.exceptions.RequestException as e:
        log.error("Network error during historical fetch: %s", e)
        return None
    except ValueError as e:
        log.error("Invalid JSON response from API: %s", e)
        return None


# ---------------------------------------------------------------------------
# Transform + Store
# ---------------------------------------------------------------------------
def _insert_dataframe(df: pd.DataFrame, db_path: str) -> int:
    """Idempotent insert -- duplicate timestamps (same PK) are silently skipped."""
    if df.empty:
        return 0
    with DB_LOCK:
        con = duckdb.connect(db_path)
        try:
            before = con.execute(f"SELECT count(*) FROM {TABLE_NAME}").fetchone()[0]
            con.execute(
                f"INSERT INTO {TABLE_NAME} SELECT * FROM df ON CONFLICT DO NOTHING"
            )
            after = con.execute(f"SELECT count(*) FROM {TABLE_NAME}").fetchone()[0]
            return after - before
        finally:
            con.close()


def transform_and_append_current(payload: dict, db_path: str = DB_PATH) -> None:
    """Extract the 'current' reading and append it as one row."""
    try:
        current = payload.get("current", {})
        ts = current.get("time")
        record = {
            "utc_timestamp": pd.to_datetime(ts) if ts else datetime.now(timezone.utc),
            "latitude": payload.get("latitude"),
            "longitude": payload.get("longitude"),
            **{field: current.get(field) for field in POLLUTANT_FIELDS},
        }
        df = pd.DataFrame([record])
        inserted = _insert_dataframe(df, db_path)
        if inserted:
            log.info("Recorded live reading at %s", record["utc_timestamp"])
        else:
            log.info("Live reading at %s already present, skipped.", record["utc_timestamp"])
    except Exception as e:
        log.error("Error processing/writing current payload: %s", e)


def transform_and_append_historical(payload: dict, db_path: str = DB_PATH) -> int:
    """Extract the 'hourly' block and bulk-insert every past hour."""
    try:
        hourly = payload.get("hourly", {})
        times = hourly.get("time", [])
        if not times:
            return 0
        rows = {
            "utc_timestamp": pd.to_datetime(times),
            "latitude": [payload.get("latitude")] * len(times),
            "longitude": [payload.get("longitude")] * len(times),
        }
        for field in POLLUTANT_FIELDS:
            rows[field] = hourly.get(field, [None] * len(times))
        df = pd.DataFrame(rows).dropna(subset=["pm2_5"], how="all")
        inserted = _insert_dataframe(df, db_path)
        log.info("Backfilled %d historical hourly rows.", inserted)
        return inserted
    except Exception as e:
        log.error("Error processing/writing historical payload: %s", e)
        return 0


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def backfill_historical(
    db_path: str = DB_PATH,
    latitude: float = LATITUDE,
    longitude: float = LONGITUDE,
    past_days: int = 3,
) -> int:
    """One-time seed of recent history so the dashboard has data to chart
    immediately, instead of waiting hours for the live poller to accumulate it."""
    init_db(db_path)
    payload = fetch_historical_payload(latitude, longitude, past_days)
    if payload:
        return transform_and_append_historical(payload, db_path)
    log.warning("Historical backfill skipped -- API fetch failed.")
    return 0


def run_ingestion_cycle(
    db_path: str = DB_PATH, latitude: float = LATITUDE, longitude: float = LONGITUDE
) -> bool:
    """Single fetch-and-store cycle. Returns True on success."""
    init_db(db_path)
    payload = fetch_current_payload(latitude, longitude)
    if payload:
        transform_and_append_current(payload, db_path)
        return True
    log.warning("Skipping DB write -- live API fetch failed.")
    return False


def run_ingestion_loop(
    db_path: str = DB_PATH,
    interval: int = POLL_INTERVAL_SECONDS,
    latitude: float = LATITUDE,
    longitude: float = LONGITUDE,
    stop_event: Optional[threading.Event] = None,
) -> None:
    """Resilient scheduled polling loop. If `stop_event` is given, the loop
    exits cleanly when it's set (used by the Streamlit background thread)."""
    log.info("Starting air-quality ingestion service. Target interval: %ss", interval)
    init_db(db_path)
    backfill_historical(db_path, latitude, longitude)

    while stop_event is None or not stop_event.is_set():
        start_time = time.time()

        run_ingestion_cycle(db_path, latitude, longitude)

        elapsed = time.time() - start_time
        sleep_duration = max(0.0, interval - elapsed)
        log.info("Cycle complete. Sleeping for %.2fs...", sleep_duration)

        # Sleep in small increments so a stop_event is honored promptly.
        slept = 0.0
        while slept < sleep_duration:
            if stop_event is not None and stop_event.is_set():
                return
            chunk = min(1.0, sleep_duration - slept)
            time.sleep(chunk)
            slept += chunk


if __name__ == "__main__":
    try:
        run_ingestion_loop()
    except KeyboardInterrupt:
        log.info("Ingestion service manually stopped by user. Exiting safely.")
