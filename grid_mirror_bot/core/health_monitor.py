"""
Health Monitor — Week 4 Core Component.
Watchdog that monitors all PairWorker components every 60 seconds.
Alerts via WhatsApp and attempts restart of unresponsive components.
"""

import asyncio
import logging
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict

# ── Logger setup ────────────────────────────────────────────────────
_logs_dir = Path(__file__).parent.parent / "logs"
_logs_dir.mkdir(parents=True, exist_ok=True)
_log_file = _logs_dir / "health_monitor.log"

logger = logging.getLogger("health_monitor")
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    _handler = RotatingFileHandler(_log_file, maxBytes=10 * 1024 * 1024, backupCount=5)
    _formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    _handler.setFormatter(_formatter)
    logger.addHandler(_handler)

COMPONENT_TIMEOUT_SECONDS = 120


class HealthMonitor:
    """Monitors all PairWorker components and alerts on unresponsive state.

    Args:
        pair_workers: Dict of PairWorker instances keyed by Binance pair.
        bot_state: Global BotState instance.
        telegram: WhatsApp/Telegram alerter.
    """

    def __init__(
        self,
        pair_workers: Dict[str, Any],
        bot_state: Any,
        telegram: Any,
    ) -> None:
        self.pair_workers = pair_workers
        self.bot_state = bot_state
        self.telegram = telegram
        self._task: asyncio.Task | None = None
        self._check_interval = 60

    async def start(self) -> None:
        """Starts the health monitoring loop."""
        logger.info(
            f"Health monitor started — checking {len(self.pair_workers)} pairs "
            f"every {self._check_interval}s"
        )
        while True:
            try:
                await self._check_health()
            except asyncio.CancelledError:
                logger.info("Health monitor cancelled.")
                raise
            except Exception as e:
                logger.error(f"Error in health monitor: {e}", exc_info=True)
            await asyncio.sleep(self._check_interval)

    async def _check_health(self) -> None:
        """Checks all components for each pair worker."""
        now = time.time()
        issues: list = []

        for pair, worker in self.pair_workers.items():
            pair_state = worker.pair_state

            if pair_state.watcher_last_seen > 0:
                elapsed = now - pair_state.watcher_last_seen
                if elapsed > COMPONENT_TIMEOUT_SECONDS:
                    pair_state.watcher_alive = False
                    issues.append((pair, "watcher", elapsed))

            if pair_state.regime_detector_last_seen > 0:
                elapsed = now - pair_state.regime_detector_last_seen
                if elapsed > COMPONENT_TIMEOUT_SECONDS:
                    pair_state.regime_detector_alive = False
                    issues.append((pair, "regime_detector", elapsed))

            if pair_state.crash_monitor_last_seen > 0:
                elapsed = now - pair_state.crash_monitor_last_seen
                if elapsed > COMPONENT_TIMEOUT_SECONDS:
                    pair_state.crash_monitor_alive = False
                    issues.append((pair, "crash_monitor", elapsed))

            if pair_state.reconciler_last_seen > 0:
                elapsed = now - pair_state.reconciler_last_seen
                if elapsed > COMPONENT_TIMEOUT_SECONDS:
                    pair_state.reconciler_alive = False
                    issues.append((pair, "reconciler", elapsed))

        if issues:
            for pair, component, elapsed in issues:
                logger.critical(
                    f"Component {component} for {pair} unresponsive — "
                    f"last seen {elapsed:.0f}s ago"
                )
                await self.telegram.send(
                    f"HEALTH ALERT: {component} for {pair} unresponsive "
                    f"(last seen {elapsed:.0f}s ago)"
                )
                await self._attempt_restart(pair, component)
        else:
            logger.debug("All components healthy")

    async def _attempt_restart(self, pair: str, component: str) -> None:
        """Attempts to restart a specific component for a pair.

        Args:
            pair: Binance pair name.
            component: Component name to restart.
        """
        worker = self.pair_workers.get(pair)
        if worker is None:
            logger.error(f"Cannot restart {component} — worker for {pair} not found")
            return

        logger.info(f"Attempting restart of {component} for {pair}")

        try:
            if component == "regime_detector" and worker.regime_detector:
                for task in worker._tasks:
                    if task.get_name() == f"regime_{pair}":
                        task.cancel()
                        try:
                            await task
                        except asyncio.CancelledError:
                            pass
                        worker._tasks.remove(task)
                        break
                new_task = asyncio.create_task(
                    worker.regime_detector.start(), name=f"regime_{pair}"
                )
                worker._tasks.append(new_task)
                logger.info(f"Restarted regime_detector for {pair}")

            elif component == "crash_monitor" and worker.crash_monitor:
                for task in worker._tasks:
                    if task.get_name() == f"crash_{pair}":
                        task.cancel()
                        try:
                            await task
                        except asyncio.CancelledError:
                            pass
                        worker._tasks.remove(task)
                        break
                new_task = asyncio.create_task(
                    worker.crash_monitor.start(), name=f"crash_{pair}"
                )
                worker._tasks.append(new_task)
                logger.info(f"Restarted crash_monitor for {pair}")

            elif component == "reconciler" and worker.reconciler:
                for task in worker._tasks:
                    if task.get_name() == f"reconciler_{pair}":
                        task.cancel()
                        try:
                            await task
                        except asyncio.CancelledError:
                            pass
                        worker._tasks.remove(task)
                        break
                new_task = asyncio.create_task(
                    worker.reconciler.start(), name=f"reconciler_{pair}"
                )
                worker._tasks.append(new_task)
                logger.info(f"Restarted reconciler for {pair}")

            elif component == "watcher":
                logger.warning(
                    f"Cannot restart watcher for {pair} — "
                    f"watcher is shared across all pairs"
                )

            await self.telegram.send(
                f"Restarted {component} for {pair}"
            )

        except Exception as e:
            logger.error(f"Failed to restart {component} for {pair}: {e}", exc_info=True)
            await self.telegram.send(
                f"FAILED to restart {component} for {pair}: {e}"
            )
