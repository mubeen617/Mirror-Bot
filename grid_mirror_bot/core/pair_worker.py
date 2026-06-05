"""
Pair Worker — Week 4 Core Component.
Central orchestrator for a single trading pair. One PairWorker per active pair.
Owns and manages per-pair compounding, regime detection, crash monitoring,
and reconciliation components.
"""

import asyncio
import logging
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.pair_state import PairState
from core.compounding import CompoundingEngine
from core.regime_detector import RegimeDetector
from core.crash_monitor import CrashMonitor
from core.reconciler import Reconciler

# ── Logger factory ──────────────────────────────────────────────────
_logs_dir = Path(__file__).parent.parent / "logs"
_logs_dir.mkdir(parents=True, exist_ok=True)


def _create_pair_logger(pair: str) -> logging.Logger:
    """Creates a dedicated rotating logger for a pair worker."""
    name = f"pair_worker_{pair.lower()}"
    log_file = _logs_dir / f"pair_worker_{pair.lower()}.log"

    pair_logger = logging.getLogger(name)
    pair_logger.setLevel(logging.DEBUG)
    if not pair_logger.handlers:
        handler = RotatingFileHandler(log_file, maxBytes=10 * 1024 * 1024, backupCount=5)
        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        handler.setFormatter(formatter)
        pair_logger.addHandler(handler)
    return pair_logger


class PairWorker:
    """Orchestrates all components for a single trading pair.

    Args:
        pair: Binance pair symbol (e.g. "BTCUSDT").
        pair_config: Per-pair configuration from config.yaml.
        global_config: Full application configuration dictionary.
        pair_state: PairState instance for this pair.
        mt5_executor: Shared MT5Executor instance.
        telegram: WhatsApp/Telegram alerter.
        binance_client: Authenticated AsyncClient instance.
        db_manager: Optional DatabaseManager for logging.
        all_pair_states: Dict of all pair states for capital ratio calculations.
    """

    def __init__(
        self,
        pair: str,
        pair_config: Dict[str, Any],
        global_config: Dict[str, Any],
        pair_state: PairState,
        mt5_executor: Any,
        telegram: Any,
        binance_client: Any,
        db_manager: Any = None,
        all_pair_states: Optional[Dict[str, PairState]] = None,
    ) -> None:
        self.pair = pair
        self.pair_config = pair_config
        self.global_config = global_config
        self.pair_state = pair_state
        self.executor = mt5_executor
        self.telegram = telegram
        self.client = binance_client
        self.db_manager = db_manager
        self.all_pair_states = all_pair_states or {}

        self.logger = _create_pair_logger(pair)
        self._tasks: List[asyncio.Task] = []

        regime_cfg = pair_config.get("regime", global_config.get("regime", {}))
        self._regime_config = {
            "regime": regime_cfg,
            "binance": {"symbol": pair},
        }
        self._regime_config.update({
            k: v for k, v in global_config.items()
            if k not in ("regime", "binance")
        })

        crash_cfg = pair_config.get("crash_monitor", global_config.get("crash_monitor", {}))
        self._crash_config = {
            "crash_monitor": crash_cfg,
            "binance": {"symbol": pair},
        }
        self._crash_config.update({
            k: v for k, v in global_config.items()
            if k not in ("crash_monitor", "binance")
        })

        self.compounding: Optional[CompoundingEngine] = None
        self.regime_detector: Optional[RegimeDetector] = None
        self.crash_monitor: Optional[CrashMonitor] = None
        self.reconciler: Optional[Reconciler] = None

    async def start(self) -> None:
        """Initialises and starts all per-pair components as async tasks."""
        self.logger.info(f"Starting PairWorker for {self.pair} -> {self.pair_state.mt5_symbol}")

        self.compounding = CompoundingEngine(self.global_config, self.db_manager)
        self.compounding.binance_balance = self.pair_state.grid_capital
        self.compounding.baseline_balance = self.pair_state.grid_capital

        self.regime_detector = PairRegimeDetector(
            config=self._regime_config,
            pair_state=self.pair_state,
            telegram=self.telegram,
            binance_client=self.client,
            executor=self.executor,
            pair=self.pair,
        )

        self.crash_monitor = PairCrashMonitor(
            config=self._crash_config,
            pair_state=self.pair_state,
            executor=self.executor,
            telegram=self.telegram,
            binance_client=self.client,
            pair=self.pair,
        )

        reconciler_config = dict(self.global_config)
        reconciler_config["binance"] = {"symbol": self.pair}

        self.reconciler = PairReconciler(
            config=reconciler_config,
            pair_state=self.pair_state,
            executor=self.executor,
            binance_client=self.client,
            telegram=self.telegram,
            pair=self.pair,
        )

        self._tasks = [
            asyncio.create_task(self.regime_detector.start(), name=f"regime_{self.pair}"),
            asyncio.create_task(self.crash_monitor.start(), name=f"crash_{self.pair}"),
            asyncio.create_task(self.reconciler.start(), name=f"reconciler_{self.pair}"),
            asyncio.create_task(self._compounding_poll_loop(), name=f"compounding_{self.pair}"),
        ]

        self.logger.info(f"PairWorker {self.pair} fully started — 4 tasks running")

    async def on_fill(self, fill: Dict[str, Any]) -> None:
        """Processes a fill event from PairWatcher.

        Args:
            fill: Fill dict with symbol, side, qty, price, order_id, timestamp.
        """
        self.logger.info(
            f"FILL | {self.pair} | {fill['side']} | "
            f"qty={fill['qty']} | price={fill['price']} | "
            f"order_id={fill['order_id']}"
        )

        if not self.pair_state.mt5_symbol_available:
            self.logger.warning(
                f"MT5 symbol {self.pair_state.mt5_symbol} not available — skipping fill"
            )
            return

        if not self.pair_state.is_trading_allowed():
            reason = self.pair_state.trading_blocked_reason()
            self.logger.info(f"Trading not allowed for {self.pair}: {reason} — skipping fill")
            async with self.pair_state.lock:
                self.pair_state.fills_today += 1
                self.pair_state.last_fill_at = time.time()
            return

        scaled = self.compounding.process_fill(fill)

        self.logger.info(
            f"SCALED | ratio={scaled['scale_ratio']:.2f}x | "
            f"qty={scaled['scaled_qty']:.6f} | "
            f"kill={scaled['kill_switch']} | reason={scaled['reason']}"
        )

        if scaled["kill_switch"]:
            self.logger.warning(f"Kill switch active: {scaled['reason']}")
            async with self.pair_state.lock:
                self.pair_state.fills_today += 1
                self.pair_state.last_fill_at = time.time()
            return

        scaled["mt5_symbol"] = self.pair_state.mt5_symbol
        scaled["symbol"] = self.pair_state.mt5_symbol

        mt5_result = await self.executor.execute(scaled)

        self.logger.info(
            f"MT5 RESULT | status={mt5_result.get('status')} | "
            f"order_id={mt5_result.get('mt5_order_id', 'N/A')}"
        )

        async with self.pair_state.lock:
            self.pair_state.fills_today += 1
            self.pair_state.last_fill_at = time.time()
            if mt5_result.get("status") == "FILLED":
                self.pair_state.mt5_orders_today += 1

        if self.telegram:
            await self.telegram.send_fill(fill, scaled)

        if self.db_manager:
            try:
                await self.db_manager.log_fill(
                    timestamp=fill["timestamp"],
                    symbol=fill["symbol"],
                    side=fill["side"],
                    binance_qty=fill["qty"],
                    binance_price=fill["price"],
                    order_id=fill["order_id"],
                    scale_ratio=scaled["scale_ratio"],
                    scaled_qty=scaled["scaled_qty"],
                    kill_switch=scaled["kill_switch"],
                    kill_reason=scaled["reason"] if scaled["kill_switch"] else None,
                )
            except Exception as e:
                self.logger.error(f"Failed to log fill to database: {e}")

    async def stop(self) -> None:
        """Cancels all running tasks cleanly."""
        self.logger.info(f"Stopping PairWorker for {self.pair}")
        for task in self._tasks:
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        self.logger.info(f"PairWorker {self.pair} stopped")

    def get_status(self) -> Dict[str, Any]:
        """Returns full pair state plus component health flags."""
        status = self.pair_state.to_dict()
        status["compounding"] = self.compounding.get_status() if self.compounding else {}
        return status

    async def _compounding_poll_loop(self) -> None:
        """Background task updating compounding engine state."""
        poll_interval = self.global_config.get("scaling", {}).get("poll_interval_seconds", 60)
        while True:
            try:
                await self.compounding.poll(self.client)
                async with self.pair_state.lock:
                    self.pair_state.scale_ratio = self.compounding.smoothed_ratio
                    self.pair_state.smoothed_ratio = self.compounding.smoothed_ratio
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in compounding poll: {e}", exc_info=True)
            await asyncio.sleep(poll_interval)


class PairRegimeDetector(RegimeDetector):
    """Per-pair regime detector that updates PairState instead of BotState.

    Overrides parent to write regime data to the per-pair PairState instance
    and only close positions for this specific pair's MT5 symbol.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        pair_state: PairState,
        telegram: Any,
        binance_client: Any,
        executor: Any = None,
        pair: str = "",
        db_manager: Any = None,
    ) -> None:
        from core.bot_state import BotState
        self._pair_state = pair_state
        self._pair = pair
        self._proxy_bot_state = BotState()

        super().__init__(
            config=config,
            bot_state=self._proxy_bot_state,
            telegram=telegram,
            binance_client=binance_client,
            executor=executor,
            db_manager=db_manager,
        )

    async def _classify(self) -> None:
        """Override to fetch pair-specific klines and update PairState."""
        klines = await self.client.get_klines(
            symbol=self._pair, interval="1m", limit=1440
        )
        if not klines or len(klines) < 15:
            return

        highs = [float(k[2]) for k in klines]
        lows = [float(k[3]) for k in klines]
        closes = [float(k[4]) for k in klines]

        atr = self._compute_atr(highs, lows, closes, period=14)
        slope_pct = self._compute_slope(closes, window=60)
        band = self._compute_band(highs, lows)
        regime = self._determine_regime(atr, slope_pct, band)

        self._pair_state.atr = atr
        self._pair_state.slope_pct = slope_pct
        self._pair_state.band_24h = band
        self._pair_state.current_price = closes[-1]
        self._pair_state.regime_detector_alive = True
        self._pair_state.regime_detector_last_seen = time.time()
        self._pair_state.regime_updated_at = time.time()

        if regime != self._prev_regime and self._prev_regime != "INITIAL":
            await self._on_pair_regime_change(self._prev_regime, regime)

        self._pair_state.prev_regime = self._prev_regime
        self._pair_state.regime = regime
        self._prev_regime = regime

    async def _on_pair_regime_change(self, prev: str, new: str) -> None:
        """Handles regime change for this specific pair only."""
        from core.regime_detector import REGIME_ACTIONS

        action = REGIME_ACTIONS.get(new, "Unknown")
        await self.telegram.send(
            f"[{self._pair}] Regime: {prev} -> {new}\n{action}"
        )

        if new in ("SLOW_BEAR", "TRENDING_HARD"):
            self._pair_state.mirror_enabled = False
            if self.executor is not None:
                try:
                    await self.executor.close_all_positions(
                        reason=f"regime_{new}_{self._pair}",
                        symbol=self._pair_state.mt5_symbol,
                    )
                except Exception as exc:
                    pass
        elif new in ("RANGING", "SLOW_BULL"):
            await self.telegram.send(
                f"[{self._pair}] Market recovering. Send /start to resume."
            )


class PairCrashMonitor(CrashMonitor):
    """Per-pair crash monitor that only closes this pair's MT5 positions.

    Overrides parent to update PairState and execute crash protocol
    only for the specific pair's symbol.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        pair_state: PairState,
        executor: Any,
        telegram: Any,
        binance_client: Any,
        pair: str = "",
    ) -> None:
        from core.bot_state import BotState
        self._pair_state = pair_state
        self._pair = pair
        self._proxy_bot_state = BotState()

        super().__init__(
            config=config,
            bot_state=self._proxy_bot_state,
            executor=executor,
            telegram=telegram,
            binance_client=binance_client,
        )

    async def _check(self) -> None:
        """Override to fetch pair-specific price and update PairState."""
        ticker = await self.client.get_symbol_ticker(symbol=self._pair)
        price = float(ticker["price"])
        now_ms = int(time.time() * 1000)

        self._price_window.append((now_ms, price))

        cutoff_ms = now_ms - (self._window_seconds * 1000)
        while self._price_window and self._price_window[0][0] < cutoff_ms:
            self._price_window.popleft()

        self._pair_state.current_price = price
        self._pair_state.price_updated_at = time.time()
        self._pair_state.crash_monitor_alive = True
        self._pair_state.crash_monitor_last_seen = time.time()

        if len(self._price_window) < 10:
            return

        oldest_price = self._price_window[0][1]
        drop = (oldest_price - price) / oldest_price if oldest_price > 0 else 0.0

        self._pair_state.drop_5m_pct = round(drop * 100, 3)

        threshold = self._get_threshold()

        if drop >= threshold and not self._triggered:
            self._triggered = True
            self._trigger_time = time.time()
            await self._execute_pair_crash_protocol(price, drop)

        if self._triggered and (time.time() - self._trigger_time) > 7200:
            self._triggered = False

    async def _execute_pair_crash_protocol(self, price: float, drop: float) -> None:
        """Executes crash protocol for this pair only."""
        fn_status: str
        try:
            closed = await self.executor.close_all_positions(
                reason=f"crash_protocol_{self._pair}",
                symbol=self._pair_state.mt5_symbol,
            )
            fn_status = f"CLOSED {closed} positions"
        except Exception as exc:
            fn_status = f"CLOSE FAILED: {exc}"

        self._pair_state.crash_lockout = True
        self._pair_state.mirror_enabled = False
        self._pair_state.regime = "CRASH"

        await self.telegram.send(
            f"[{self._pair}] CRASH PROTOCOL FIRED\n"
            f"Price: ${price:,.0f}\n"
            f"Drop: {drop * 100:.2f}% in 5 minutes\n"
            f"MT5 {self._pair_state.mt5_symbol}: {fn_status}\n"
            f"Pair PAUSED. Send /start to resume."
        )

    def reset(self) -> None:
        """Clears crash lockout for this pair."""
        self._triggered = False
        self._pair_state.crash_lockout = False


class PairReconciler(Reconciler):
    """Per-pair reconciler that updates PairState instead of BotState.

    Overrides parent to reconcile positions for a specific pair/symbol only.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        pair_state: PairState,
        executor: Any,
        binance_client: Any,
        telegram: Any,
        pair: str = "",
    ) -> None:
        from core.bot_state import BotState
        self._pair_state = pair_state
        self._pair = pair
        self._proxy_bot_state = BotState()

        super().__init__(
            config=config,
            bot_state=self._proxy_bot_state,
            executor=executor,
            binance_client=binance_client,
            telegram=telegram,
        )
        self.binance_symbol = pair

    async def _reconcile(self) -> None:
        """Single reconciliation pass for this pair."""
        self._pair_state.reconciler_alive = True
        self._pair_state.reconciler_last_seen = time.time()

        binance_net = await self._get_binance_net_position()
        async with self._pair_state.lock:
            self._pair_state.binance_net_position = binance_net

        mt5_net = await self._get_mt5_pair_position()

        smoothed_ratio = self._pair_state.smoothed_ratio
        if smoothed_ratio <= 0:
            return

        expected_mt5 = binance_net * smoothed_ratio
        drift = expected_mt5 - mt5_net

        async with self._pair_state.lock:
            self._pair_state.position_drift = drift
            self._pair_state.last_reconcile_at = time.time()

        if abs(drift) < self.drift_threshold:
            return

        if self._pair_state.drift_corrections_today >= self.max_corrections:
            return

        if self._pair_state.is_trading_allowed():
            side = "BUY" if drift > 0 else "SELL"
            correction_order = {
                "symbol": self._pair_state.mt5_symbol,
                "mt5_symbol": self._pair_state.mt5_symbol,
                "side": side,
                "scaled_qty": abs(drift),
                "scale_ratio": smoothed_ratio,
                "entry_price": 0.0,
                "kill_switch": False,
                "reason": "DRIFT_CORRECTION",
                "timestamp": int(time.time() * 1000),
                "order_id": f"DRIFT_{self._pair}_{int(time.time())}",
            }
            result = await self.executor.execute(correction_order)
            if result.get("status") == "FILLED":
                async with self._pair_state.lock:
                    self._pair_state.drift_corrections_today += 1

    async def _get_binance_net_position(self) -> float:
        """Gets the Binance net position for this pair's base asset."""
        try:
            base_asset = self._pair.replace("USDT", "")
            account_info = await self.binance_client.get_account()
            for asset in account_info.get("balances", []):
                if asset["asset"] == base_asset:
                    return float(asset["free"]) + float(asset["locked"])
            return 0.0
        except Exception:
            return self._pair_state.binance_net_position

    async def _get_mt5_pair_position(self) -> float:
        """Gets the MT5 net position for this pair's symbol."""
        try:
            positions = await self.executor.get_pair_positions(self._pair_state.mt5_symbol)
            net = 0.0
            for pos in positions:
                if pos.type == 0:  # ORDER_TYPE_BUY
                    net += pos.volume
                else:
                    net -= pos.volume
            return net
        except Exception:
            return self._pair_state.mt5_net_position
