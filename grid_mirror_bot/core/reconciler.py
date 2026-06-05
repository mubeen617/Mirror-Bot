"""
Position Reconciler — 60-second Drift Detection and Correction.
Compares Binance net BTC position against the expected MT5 position
(scaled by smoothed_ratio) and fires correction orders when drift
exceeds the configured threshold.
"""

import asyncio
import logging
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict

from core.bot_state import BotState

# ── Logger setup ────────────────────────────────────────────────────
logs_dir = Path(__file__).parent.parent / "logs"
logs_dir.mkdir(parents=True, exist_ok=True)
log_file = logs_dir / "reconciler.log"

reconciler_logger = logging.getLogger("reconciler")
reconciler_logger.setLevel(logging.DEBUG)
if not reconciler_logger.handlers:
    handler = RotatingFileHandler(log_file, maxBytes=10 * 1024 * 1024, backupCount=5)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    handler.setFormatter(formatter)
    reconciler_logger.addHandler(handler)


class Reconciler:
    """Periodically compares Binance and MT5 positions, correcting drift.

    Args:
        config: Full application configuration dictionary.
        bot_state: Shared ``BotState`` instance.
        executor: ``MT5Executor`` used to fire correction orders.
        binance_client: ``AsyncClient`` for querying Binance balances.
        telegram: ``TelegramAlerter`` for drift notifications.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        bot_state: BotState,
        executor: Any,
        binance_client: Any,
        telegram: Any,
    ) -> None:
        self.config = config
        self.bot_state = bot_state
        self.executor = executor
        self.binance_client = binance_client
        self.telegram = telegram

        recon_cfg = config.get("reconciler", {})
        self.interval_seconds = int(recon_cfg.get("interval_seconds", 60))
        self.drift_threshold = float(recon_cfg.get("drift_threshold_btc", 0.001))
        self.max_corrections = int(recon_cfg.get("max_corrections_per_day", 10))

        binance_cfg = config.get("binance", {})
        self.binance_symbol = str(binance_cfg.get("symbol", "BTCUSDT"))

        self._task: asyncio.Task[None] | None = None

    # ────────────────────────────────────────────────────────────────
    # Lifecycle
    # ────────────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Starts the reconciliation background loop."""
        recon_cfg = self.config.get("reconciler", {})
        if not recon_cfg.get("enabled", True):
            reconciler_logger.info("Reconciler is disabled in configuration. Skipping start.")
            return
        self._task = asyncio.create_task(self._loop())
        reconciler_logger.info(
            f"Reconciler started — interval: {self.interval_seconds}s | "
            f"drift threshold: {self.drift_threshold} BTC"
        )

    async def stop(self) -> None:
        """Cancels the reconciliation loop."""
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        reconciler_logger.info("Reconciler stopped.")

    async def _loop(self) -> None:
        """Runs ``_reconcile()`` every ``interval_seconds``."""
        while True:
            try:
                await self._reconcile()
            except asyncio.CancelledError:
                break
            except Exception as e:
                reconciler_logger.error(f"Error in reconcile loop: {e}", exc_info=True)
            await asyncio.sleep(self.interval_seconds)

    # ────────────────────────────────────────────────────────────────
    # Core reconciliation
    # ────────────────────────────────────────────────────────────────

    async def _reconcile(self) -> None:
        """Single reconciliation pass — compare positions and correct drift."""
        # 1. Get Binance net BTC position
        binance_net = await self._get_binance_net_position()
        async with self.bot_state.lock:
            self.bot_state.binance_net_position = binance_net

        # 2. Get MT5 net position
        mt5_net = await self.executor.get_net_position()

        # 3. Calculate expected MT5 position
        smoothed_ratio = self.bot_state.smoothed_ratio
        if smoothed_ratio <= 0:
            reconciler_logger.debug(
                "Smoothed ratio is zero — skipping reconcile."
            )
            return

        expected_mt5 = binance_net * smoothed_ratio

        # 4. Calculate drift
        drift = expected_mt5 - mt5_net
        async with self.bot_state.lock:
            self.bot_state.position_drift = drift

        reconciler_logger.debug(
            f"Reconcile — binance_net: {binance_net:.6f} | mt5_net: {mt5_net:.6f} | "
            f"expected_mt5: {expected_mt5:.6f} | drift: {drift:.6f}"
        )

        # 5. Check drift threshold
        if abs(drift) < self.drift_threshold:
            reconciler_logger.debug("Positions synced — drift within threshold.")
            await self._update_timestamp()
            return

        # 6. Drift exceeds threshold
        reconciler_logger.warning(
            f"POSITION DRIFT DETECTED — binance_net: {binance_net:.6f} | "
            f"mt5_net: {mt5_net:.6f} | expected_mt5: {expected_mt5:.6f} | "
            f"drift: {drift:.6f}"
        )
        await self.telegram.send(
            f"⚠️ Position drift detected — drift: {drift:.6f} BTC-equiv\n"
            f"Binance net: {binance_net:.6f} | MT5 net: {mt5_net:.6f} | "
            f"Expected: {expected_mt5:.6f}"
        )

        # Check max corrections
        if self.bot_state.drift_corrections_today >= self.max_corrections:
            reconciler_logger.critical(
                f"Max drift corrections per day reached ({self.max_corrections}) — "
                "skipping correction."
            )
            await self.telegram.send(
                f"🚨 CRITICAL: Max drift corrections per day reached "
                f"({self.max_corrections}) — manual intervention required."
            )
            await self._update_timestamp()
            return

        # Attempt correction
        if self.bot_state.is_trading_allowed():
            side = "BUY" if drift > 0 else "SELL"
            correction_qty = abs(drift)

            correction_order = {
                "symbol": self.executor.symbol,
                "side": side,
                "scaled_qty": correction_qty,
                "scale_ratio": smoothed_ratio,
                "entry_price": 0.0,
                "kill_switch": False,
                "reason": "DRIFT_CORRECTION",
                "timestamp": int(time.time() * 1000),
                "order_id": f"DRIFT_{int(time.time())}",
            }

            reconciler_logger.info(
                f"Firing drift correction — {side} {correction_qty:.6f} {self.executor.symbol}"
            )
            result = await self.executor.execute(correction_order)

            if result.get("status") == "FILLED":
                async with self.bot_state.lock:
                    self.bot_state.drift_corrections_today += 1
                reconciler_logger.info(
                    f"Drift correction FILLED — ticket: {result.get('mt5_order_id')}"
                )
            else:
                reconciler_logger.warning(
                    f"Drift correction not filled — status: {result.get('status')} | "
                    f"reason: {result.get('reason', result.get('comment', 'unknown'))}"
                )
        else:
            reason = self.bot_state.trading_blocked_reason()
            reconciler_logger.info(
                f"Drift detected but trading not allowed — skipping correction "
                f"(reason: {reason})"
            )

        await self._update_timestamp()

    # ────────────────────────────────────────────────────────────────
    # Helpers
    # ────────────────────────────────────────────────────────────────

    async def _get_binance_net_position(self) -> float:
        """Queries the Binance account for the net BTC balance.

        Returns:
            float: Net BTC balance (free + locked).
        """
        try:
            account_info = await self.binance_client.get_account()
            btc_balance = 0.0
            for asset in account_info.get("balances", []):
                if asset["asset"] == "BTC":
                    btc_balance = float(asset["free"]) + float(asset["locked"])
                    break
            return btc_balance
        except Exception as e:
            reconciler_logger.error(f"Failed to query Binance BTC balance: {e}", exc_info=True)
            return self.bot_state.binance_net_position

    async def _update_timestamp(self) -> None:
        """Updates the last-reconcile timestamp in bot_state."""
        async with self.bot_state.lock:
            self.bot_state.last_reconcile_timestamp = int(time.time() * 1000)
