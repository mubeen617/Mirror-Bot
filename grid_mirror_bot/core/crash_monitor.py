"""
Crash Monitor — Week 3 Core Component.

The fastest and most critical safety component.  Polls BTC price every
5 seconds and compares against a rolling window.  If the price drops
by more than the configured threshold within the window, it immediately:

    1. Closes all FundedNext MT5 positions
    2. Locks the mirror bot
    3. Sets the regime to ``CRASH``
    4. Sends a critical Telegram alert

The crash lockout is only cleared by an explicit ``/start`` command.
A 120-minute lockout prevents repeated triggers during extended drops.
"""

import asyncio
import collections
import logging
import time
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Deque, Dict, Optional, Tuple

from core.bot_state import BotState

# ── Logger setup ────────────────────────────────────────────────────
_logs_dir = Path(__file__).parent.parent / "logs"
_logs_dir.mkdir(parents=True, exist_ok=True)
_log_file = _logs_dir / "crash_monitor.log"

logger = logging.getLogger("crash_monitor")
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    _handler = RotatingFileHandler(_log_file, maxBytes=10 * 1024 * 1024, backupCount=5)
    _formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    _handler.setFormatter(_formatter)
    logger.addHandler(_handler)


class CrashMonitor:
    """Monitors BTC for sudden price drops and executes crash protocol.

    Args:
        config: Full application configuration dictionary.
        bot_state: Shared ``BotState`` instance.
        executor: ``MT5Executor`` for emergency position closure.
        telegram: ``TelegramAlerter`` for critical notifications.
        binance_client: ``binance.AsyncClient`` for price ticker queries.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        bot_state: BotState,
        executor: Any,
        telegram: Any,
        binance_client: Any,
    ) -> None:
        self.config = config
        self.bot_state = bot_state
        self.executor = executor
        self.telegram = telegram
        self.client = binance_client

        crash_cfg = config.get("crash_monitor", {})
        self._drop_threshold = float(crash_cfg.get("drop_threshold_pct", 0.015))
        self._window_seconds = int(crash_cfg.get("window_seconds", 300))
        self._poll_interval = int(crash_cfg.get("poll_interval_seconds", 5))
        self._weekend_threshold = float(crash_cfg.get("weekend_drop_threshold", 0.020))
        self._require_manual = bool(crash_cfg.get("require_manual_restart", True))

        self._price_window: Deque[Tuple[int, float]] = collections.deque()
        self._triggered: bool = False
        self._trigger_time: float = 0.0

    # ────────────────────────────────────────────────────────────────
    # Public API
    # ────────────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Runs the crash-check loop forever.

        Each iteration sleeps ``poll_interval_seconds`` (5 s) then
        calls ``_check()``.  Exceptions are logged at ERROR and never
        propagate — the loop always continues.
        """
        logger.info(
            "Crash monitor started — polling every %d seconds, "
            "threshold %.1f%%, window %ds",
            self._poll_interval,
            self._drop_threshold * 100,
            self._window_seconds,
        )
        while True:
            try:
                await self._check()
            except asyncio.CancelledError:
                logger.info("Crash monitor cancelled.")
                raise
            except Exception as exc:
                logger.error("Error in crash monitor: %s", exc, exc_info=True)
            await asyncio.sleep(self._poll_interval)

    def reset(self) -> None:
        """Clears crash lockout — called by ``/start`` Telegram command.

        Note:
            ``mirror_enabled`` stays False — the regime detector must
            confirm RANGING or SLOW_BULL before mirror is re-enabled.
        """
        self._triggered = False
        self.bot_state.crash_lockout = False
        logger.info("Crash monitor reset — awaiting regime confirmation")

    # ────────────────────────────────────────────────────────────────
    # Price check
    # ────────────────────────────────────────────────────────────────

    async def _check(self) -> None:
        """Fetches current BTC price, updates window, and evaluates drop."""
        # 1. Fetch current price
        ticker = await self.client.get_symbol_ticker(symbol="BTCUSDT")
        price = float(ticker["price"])
        now_ms = int(time.time() * 1000)

        # 2. Append to window
        self._price_window.append((now_ms, price))

        # 3. Drop entries older than window
        cutoff_ms = now_ms - (self._window_seconds * 1000)
        while self._price_window and self._price_window[0][0] < cutoff_ms:
            self._price_window.popleft()

        # 4. Update bot_state
        self.bot_state.btc_price = price

        # 5. Need at least 10 entries
        if len(self._price_window) < 10:
            logger.debug(
                "Price: $%.0f | window_size: %d (waiting for 10+)",
                price, len(self._price_window),
            )
            return

        # 6. Get oldest price
        oldest_price = self._price_window[0][1]

        # 7. Calculate drop
        drop = (oldest_price - price) / oldest_price if oldest_price > 0 else 0.0

        # 8. Update drop metric
        self.bot_state.drop_5m_pct = round(drop * 100, 3)

        logger.debug(
            "Price: $%.0f | drop_5m: %.3f%% | window_size: %d",
            price, drop * 100, len(self._price_window),
        )

        # 9. Determine threshold (weekday vs weekend)
        threshold = self._get_threshold()

        # 10. Check trigger
        if drop >= threshold and not self._triggered:
            self._triggered = True
            self._trigger_time = time.time()
            await self._execute_crash_protocol(price, drop)

        # 11. Reset trigger lockout after 120 minutes
        if self._triggered and (time.time() - self._trigger_time) > 7200:
            self._triggered = False
            logger.info("Crash lockout reset after 120 minutes")

    # ────────────────────────────────────────────────────────────────
    # Crash protocol
    # ────────────────────────────────────────────────────────────────

    async def _execute_crash_protocol(self, price: float, drop: float) -> None:
        """Executes the emergency crash protocol in strict order.

        Steps executed sequentially:
            1. Close FundedNext positions (most urgent)
            2. Lock mirror bot
            3. Set regime to CRASH
            4. Send Telegram alert
            5. Log at CRITICAL

        Args:
            price: Current BTC price.
            drop: Drop fraction (e.g. 0.016 = 1.6%).
        """
        # Step 1 — Close FundedNext positions FIRST
        fn_status: str
        try:
            closed = await self.executor.close_all_positions(reason="crash_protocol")
            fn_status = f"CLOSED {closed} positions"
        except Exception as exc:
            fn_status = f"CLOSE FAILED: {exc}"
            logger.critical("Failed to close positions during crash protocol: %s", exc)

        # Step 2 — Lock mirror bot
        self.bot_state.crash_lockout = True
        self.bot_state.mirror_enabled = False

        # Step 3 — Set regime to CRASH
        self.bot_state.regime = "CRASH"

        # Step 4 — Send Telegram alert
        await self.telegram.send(
            f"🚨 CRASH PROTOCOL FIRED\n"
            f"BTC: ${price:,.0f}\n"
            f"Drop: {drop * 100:.2f}% in 5 minutes\n"
            f"FundedNext: {fn_status}\n"
            f"Grid: PAUSED\n"
            f"Send /start to resume after confirming recovery."
        )

        # Step 5 — Log at CRITICAL
        logger.critical(
            "CRASH PROTOCOL FIRED — BTC: $%.0f | drop: %.2f%% | FN: %s",
            price, drop * 100, fn_status,
        )

    def _get_threshold(self) -> float:
        """Returns the appropriate drop threshold based on day of week.

        Returns:
            Weekend threshold on Saturday/Sunday UTC, weekday threshold otherwise.
        """
        now_utc = datetime.now(timezone.utc)
        # Saturday=5, Sunday=6
        if now_utc.weekday() in (5, 6):
            return self._weekend_threshold
        return self._drop_threshold
