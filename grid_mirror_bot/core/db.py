"""
Database module for the grid mirror bot.
Handles SQLite schema creation and asynchronous database interactions using aiosqlite.
Uses a persistent connection to avoid repeated open/close overhead.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import aiosqlite

logger = logging.getLogger("db")


class DatabaseManager:
    """Manages SQLite database connections and execution of queries asynchronously."""

    def __init__(self, db_path: Path) -> None:
        """
        Initializes the DatabaseManager.

        Args:
            db_path (Path): Path to the SQLite database file.
        """
        self.db_path = db_path
        self._conn: Optional[aiosqlite.Connection] = None

    async def _get_conn(self) -> aiosqlite.Connection:
        """Returns the persistent connection, reconnecting if needed."""
        if self._conn is None:
            self._conn = await aiosqlite.connect(self.db_path)
            await self._conn.execute("PRAGMA journal_mode=WAL")
        return self._conn

    async def initialize(self) -> None:
        """
        Creates the database directory and tables if they do not exist.
        """
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        db = await self._get_conn()
        await db.execute("""
            CREATE TABLE IF NOT EXISTS fills (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp       INTEGER NOT NULL,
                datetime        TEXT NOT NULL,
                symbol          TEXT NOT NULL,
                side            TEXT NOT NULL,
                binance_qty     REAL NOT NULL,
                binance_price   REAL NOT NULL,
                order_id        TEXT NOT NULL,
                scale_ratio     REAL NOT NULL,
                scaled_qty      REAL NOT NULL,
                kill_switch     INTEGER NOT NULL DEFAULT 0,
                kill_reason     TEXT,
                created_at      TEXT DEFAULT CURRENT_TIMESTAMP
            );
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS regime_log (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp       INTEGER NOT NULL,
                datetime        TEXT NOT NULL,
                regime          TEXT NOT NULL,
                atr             REAL,
                slope_pct       REAL,
                band_24h        REAL,
                prev_regime     TEXT,
                created_at      TEXT DEFAULT CURRENT_TIMESTAMP
            );
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS ratio_log (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp       INTEGER NOT NULL,
                binance_balance REAL NOT NULL,
                fn_equity       REAL NOT NULL,
                raw_ratio       REAL NOT NULL,
                smoothed_ratio  REAL NOT NULL,
                created_at      TEXT DEFAULT CURRENT_TIMESTAMP
            );
        """)
        await db.commit()

    async def close(self) -> None:
        """Closes the persistent database connection."""
        if self._conn is not None:
            await self._conn.close()
            self._conn = None

    async def log_fill(
        self,
        timestamp: int,
        symbol: str,
        side: str,
        binance_qty: float,
        binance_price: float,
        order_id: str,
        scale_ratio: float,
        scaled_qty: float,
        kill_switch: bool,
        kill_reason: str | None,
    ) -> None:
        """
        Logs a fill event into the fills table.
        """
        dt_str = datetime.fromtimestamp(timestamp / 1000.0).isoformat()
        db = await self._get_conn()
        await db.execute(
            """
            INSERT INTO fills (
                timestamp, datetime, symbol, side, binance_qty,
                binance_price, order_id, scale_ratio, scaled_qty,
                kill_switch, kill_reason
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                timestamp,
                dt_str,
                symbol,
                side,
                binance_qty,
                binance_price,
                str(order_id),
                scale_ratio,
                scaled_qty,
                1 if kill_switch else 0,
                kill_reason,
            ),
        )
        await db.commit()

    async def log_regime(
        self,
        timestamp: int,
        regime: str,
        atr: float | None,
        slope_pct: float | None,
        band_24h: float | None,
        prev_regime: str | None,
    ) -> None:
        """
        Logs a market regime reading.
        """
        dt_str = datetime.fromtimestamp(timestamp).isoformat()
        db = await self._get_conn()
        await db.execute(
            """
            INSERT INTO regime_log (
                timestamp, datetime, regime, atr, slope_pct, band_24h, prev_regime
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (timestamp, dt_str, regime, atr, slope_pct, band_24h, prev_regime),
        )
        await db.commit()

    async def log_ratio(
        self,
        timestamp: int,
        binance_balance: float,
        fn_equity: float,
        raw_ratio: float,
        smoothed_ratio: float,
    ) -> None:
        """
        Logs a scaling ratio reading.
        """
        db = await self._get_conn()
        await db.execute(
            """
            INSERT INTO ratio_log (
                timestamp, binance_balance, fn_equity, raw_ratio, smoothed_ratio
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (timestamp, binance_balance, fn_equity, raw_ratio, smoothed_ratio),
        )
        await db.commit()
