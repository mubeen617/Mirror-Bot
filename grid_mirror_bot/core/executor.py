"""
MT5 Order Executor — Week 2 Core Component.
Connects to MetaTrader 5 demo account, places mirrored orders,
monitors connection health, and enforces drawdown kill switches.
"""

import asyncio
import logging
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None  # type: ignore[assignment]  # Will be mocked in tests

from core.bot_state import BotState

# ── Logger setup ────────────────────────────────────────────────────
logs_dir = Path(__file__).parent.parent / "logs"
logs_dir.mkdir(parents=True, exist_ok=True)
log_file = logs_dir / "executor.log"

executor_logger = logging.getLogger("executor")
executor_logger.setLevel(logging.DEBUG)
if not executor_logger.handlers:
    handler = RotatingFileHandler(log_file, maxBytes=10 * 1024 * 1024, backupCount=5)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    handler.setFormatter(formatter)
    executor_logger.addHandler(handler)


class MT5Executor:
    """Manages the MetaTrader 5 connection, order execution, and drawdown enforcement.

    Args:
        config: Full application configuration dictionary.
        bot_state: Shared ``BotState`` instance.
        telegram: ``TelegramAlerter`` for sending critical notifications.
    """

    def __init__(self, config: Dict[str, Any], bot_state: BotState, telegram: Any) -> None:
        self.config = config
        self.bot_state = bot_state
        self.telegram = telegram

        mt5_cfg = config.get("mt5", {})
        self.login = int(mt5_cfg.get("login", 0))
        self.password = str(mt5_cfg.get("password", ""))
        self.server = str(mt5_cfg.get("server", ""))
        self.symbol = str(mt5_cfg.get("symbol", "BTCUSD"))
        self.deviation = int(mt5_cfg.get("deviation", 20))
        self.path = str(mt5_cfg.get("path", ""))
        if not self.path:
            # Check default IC Markets and generic MT5 paths
            default_paths = [
                "C:/Program Files/MetaTrader 5 IC Markets Global/terminal64.exe",
                "C:/Program Files/MetaTrader 5/terminal64.exe",
            ]
            for p in default_paths:
                if Path(p).exists():
                    self.path = p
                    break
        if self.path:
            executor_logger.info(f"Resolved MT5 terminal path: {self.path}")
        self.magic_number = int(mt5_cfg.get("magic_number", 20260001))
        self.demo_mode = bool(mt5_cfg.get("demo_mode", True))
        self.connection_timeout = int(mt5_cfg.get("connection_timeout_seconds", 30))
        self.reconnect_interval = int(mt5_cfg.get("reconnect_interval_seconds", 10))
        self.max_reconnect_attempts = int(mt5_cfg.get("max_reconnect_attempts", 10))

        risk_cfg = config.get("risk", {})
        self.daily_loss_limit_pct = float(risk_cfg.get("daily_loss_limit_pct", 0.04))
        self.max_drawdown_pct = float(risk_cfg.get("max_drawdown_pct", 0.10))
        self.safety_margin = float(risk_cfg.get("safety_margin", 0.80))

        self.day_open_equity: float = 0.0
        self._monitor_task: Optional[asyncio.Task[None]] = None
        self._reconnect_count: int = 0

    # ────────────────────────────────────────────────────────────────
    # Connection lifecycle
    # ────────────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Initialises MT5 terminal connection and starts the health monitor.

        Raises:
            RuntimeError: If MT5 initialisation or login fails.
        """
        await self._connect()
        self._monitor_task = asyncio.create_task(self._connection_monitor())
        executor_logger.info("MT5 executor started — connection monitor active.")

    async def stop(self) -> None:
        """Shuts down the MT5 connection and cancels the monitor task."""
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        mt5.shutdown()
        async with self.bot_state.lock:
            self.bot_state.mt5_connected = False
        executor_logger.info("MT5 executor stopped.")

    async def _connect(self) -> None:
        """Performs MT5 initialisation and account verification."""
        loop = asyncio.get_running_loop()
        
        init_kwargs = {"timeout": self.connection_timeout * 1000}
        if self.path:
            init_kwargs["path"] = self.path
            executor_logger.info(f"Attempting to bind to active running MT5 terminal at path: {self.path}...")
        else:
            executor_logger.info("Attempting to bind to active running MT5 terminal...")

        init_ok = await loop.run_in_executor(
            None,
            lambda: mt5.initialize(**init_kwargs),
        )

        if not init_ok:
            executor_logger.warning("No-args MT5 initialization failed. Retrying with full credentials...")
            full_init_kwargs = {
                "login": self.login,
                "password": self.password,
                "server": self.server,
                "timeout": self.connection_timeout * 1000,
            }
            if self.path:
                full_init_kwargs["path"] = self.path
            init_ok = await loop.run_in_executor(
                None,
                lambda: mt5.initialize(**full_init_kwargs),
            )

        if not init_ok:
            error = mt5.last_error()
            msg = f"MT5 initialize() failed — error: {error}"
            executor_logger.error(msg)
            raise RuntimeError(msg)

        account = await loop.run_in_executor(None, mt5.account_info)
        if account is None:
            error = mt5.last_error()
            msg = f"MT5 account_info() returned None — error: {error}"
            executor_logger.error(msg)
            raise RuntimeError(msg)

        # Populate state
        async with self.bot_state.lock:
            self.bot_state.mt5_connected = True
            self.bot_state.mt5_equity = account.equity
            if self.bot_state.mt5_peak_equity == 0.0:
                self.bot_state.mt5_peak_equity = account.equity

        if self.day_open_equity == 0.0:
            self.day_open_equity = account.equity

        executor_logger.info(
            f"MT5 connected — login: {account.login} | server: {account.server} | "
            f"equity: ${account.equity:,.2f} | currency: {account.currency} | "
            f"leverage: 1:{account.leverage}"
        )
        mode_str = "demo" if self.demo_mode else "live"
        await self.telegram.send(
            f"✅ MT5 connected — {mode_str} account — equity ${account.equity:,.2f}"
        )

    async def _connection_monitor(self) -> None:
        """Background loop polling MT5 account health every 10 seconds."""
        while True:
            try:
                await asyncio.sleep(self.reconnect_interval)
                loop = asyncio.get_running_loop()
                account = await loop.run_in_executor(None, mt5.account_info)

                if account is None:
                    executor_logger.warning("MT5 connection lost — account_info() returned None.")
                    async with self.bot_state.lock:
                        self.bot_state.mt5_connected = False
                    await self._reconnect()
                else:
                    async with self.bot_state.lock:
                        self.bot_state.mt5_connected = True
                        self.bot_state.mt5_equity = account.equity
                    self._reconnect_count = 0

            except asyncio.CancelledError:
                break
            except Exception as e:
                executor_logger.error(f"Error in connection monitor: {e}", exc_info=True)

    async def _reconnect(self) -> None:
        """Attempts exponential-backoff reconnection to MT5."""
        backoff = self.reconnect_interval
        for attempt in range(1, self.max_reconnect_attempts + 1):
            executor_logger.info(f"MT5 reconnect attempt {attempt}/{self.max_reconnect_attempts} in {backoff}s...")
            await asyncio.sleep(backoff)

            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, mt5.shutdown)

            try:
                reconnect_kwargs = {
                    "login": self.login,
                    "password": self.password,
                    "server": self.server,
                    "timeout": self.connection_timeout * 1000,
                }
                if self.path:
                    reconnect_kwargs["path"] = self.path
                init_ok = await loop.run_in_executor(
                    None,
                    lambda: mt5.initialize(**reconnect_kwargs),
                )
                if init_ok:
                    account = await loop.run_in_executor(None, mt5.account_info)
                    if account is not None:
                        async with self.bot_state.lock:
                            self.bot_state.mt5_connected = True
                            self.bot_state.mt5_equity = account.equity
                        self._reconnect_count = 0
                        executor_logger.info(
                            f"MT5 reconnected successfully on attempt {attempt} — "
                            f"equity: ${account.equity:,.2f}"
                        )
                        await self.telegram.send(
                            f"✅ MT5 reconnected on attempt {attempt} — equity ${account.equity:,.2f}"
                        )
                        return
            except Exception as e:
                executor_logger.error(f"Reconnect attempt {attempt} failed: {e}")

            # Exponential backoff: 10s, 20s, 40s … max 60s
            backoff = min(60, backoff * 2)

        # Exhausted all attempts
        self._reconnect_count += 1
        executor_logger.critical(
            f"MT5 reconnection failed after {self.max_reconnect_attempts} attempts."
        )
        await self.telegram.send(
            f"🚨 CRITICAL: MT5 reconnection failed after {self.max_reconnect_attempts} attempts — "
            "executor will NOT place orders until connection is restored."
        )

    # ────────────────────────────────────────────────────────────────
    # Order execution
    # ────────────────────────────────────────────────────────────────

    async def execute(self, scaled_order: Dict[str, Any]) -> Dict[str, Any]:
        """Places a mirrored order on the MT5 demo account.

        Args:
            scaled_order: Dict from ``CompoundingEngine.process_fill()``
                containing ``symbol``, ``side``, ``scaled_qty``, ``scale_ratio``,
                ``entry_price``, ``kill_switch``, ``reason``, ``timestamp``.

        Returns:
            Dict with ``status`` (``FILLED``, ``SKIPPED``, or ``FAILED``)
            and associated details.
        """
        # 1. Trading-allowed gate
        if not self.bot_state.is_trading_allowed():
            reason = self.bot_state.trading_blocked_reason()
            executor_logger.info(f"Order SKIPPED — trading not allowed: {reason}")
            return {"status": "SKIPPED", "reason": reason}

        # 2. MT5 connectivity gate
        if not self.bot_state.mt5_connected:
            executor_logger.info("Order SKIPPED — MT5 disconnected.")
            return {"status": "SKIPPED", "reason": "MT5_DISCONNECTED"}

        # 3. Validate order
        side = scaled_order.get("side", "").upper()
        qty = float(scaled_order.get("scaled_qty", 0.0))
        symbol = scaled_order.get("symbol", self.symbol)

        if qty <= 0.0:
            executor_logger.warning(f"Order SKIPPED — invalid qty: {qty}")
            return {"status": "SKIPPED", "reason": "INVALID_QTY"}
        if side not in ("BUY", "SELL"):
            executor_logger.warning(f"Order SKIPPED — invalid side: {side}")
            return {"status": "SKIPPED", "reason": "INVALID_SIDE"}

        # 4. Fetch symbol info
        loop = asyncio.get_running_loop()
        sym_info = await loop.run_in_executor(None, lambda: mt5.symbol_info(symbol))
        if sym_info is None:
            executor_logger.error(f"Symbol not found in MT5: {symbol}")
            return {"status": "FAILED", "reason": "SYMBOL_NOT_FOUND"}

        # Ensure symbol is visible in Market Watch
        if not sym_info.visible:
            await loop.run_in_executor(None, lambda: mt5.symbol_select(symbol, True))

        # 5. Round volume to symbol step
        volume_step = sym_info.volume_step
        volume_min = sym_info.volume_min
        rounded_volume = max(
            volume_min,
            round(round(qty / volume_step) * volume_step, 8),
        )

        # 6. Get current tick
        tick = await loop.run_in_executor(None, lambda: mt5.symbol_info_tick(symbol))
        if tick is None:
            executor_logger.error(f"No tick data for {symbol}")
            return {"status": "FAILED", "reason": "NO_TICK_DATA"}

        price = tick.ask if side == "BUY" else tick.bid
        order_type = mt5.ORDER_TYPE_BUY if side == "BUY" else mt5.ORDER_TYPE_SELL

        # 7. Build order request
        fill_order_id = str(scaled_order.get("order_id", scaled_order.get("timestamp", "")))
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": rounded_volume,
            "type": order_type,
            "price": price,
            "deviation": self.deviation,
            "magic": self.magic_number,
            "comment": f"GridMirror_{fill_order_id}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        executor_logger.info(
            f"Sending MT5 order — {side} {rounded_volume} {symbol} @ {price:.2f} "
            f"(deviation: {self.deviation}, magic: {self.magic_number})"
        )

        # 8. Send order
        result = await loop.run_in_executor(None, lambda: mt5.order_send(request))

        # 9. Check result
        if result is None:
            error = mt5.last_error()
            executor_logger.error(f"MT5 order_send returned None — error: {error}")
            await self.telegram.send(f"❌ MT5 order failed — order_send returned None: {error}")
            return {"status": "FAILED", "reason": "ORDER_SEND_NONE", "error": str(error)}

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            executor_logger.error(
                f"MT5 order REJECTED — retcode: {result.retcode} | "
                f"comment: {result.comment}"
            )
            await self.telegram.send(
                f"❌ MT5 order REJECTED — retcode: {result.retcode} — {result.comment}"
            )
            return {
                "status": "FAILED",
                "retcode": result.retcode,
                "comment": result.comment,
            }

        # 10. Success — update state
        now_ms = int(time.time() * 1000)
        async with self.bot_state.lock:
            self.bot_state.mt5_orders_today += 1
            self.bot_state.last_fill_timestamp = now_ms

        executor_logger.info(
            f"MT5 order FILLED — ticket: {result.order} | {side} {rounded_volume} "
            f"{symbol} @ {result.price}"
        )
        await self.telegram.send(
            f"✅ MT5 {side} {rounded_volume} {symbol} @ ${result.price:,.2f} "
            f"(ticket: {result.order})"
        )

        return {
            "status": "FILLED",
            "mt5_order_id": result.order,
            "symbol": symbol,
            "side": side,
            "volume": rounded_volume,
            "price": result.price,
            "binance_order_id": fill_order_id,
            "scale_ratio": float(scaled_order.get("scale_ratio", 0.0)),
            "timestamp": now_ms,
        }

    # ────────────────────────────────────────────────────────────────
    # Position management
    # ────────────────────────────────────────────────────────────────

    async def close_all_positions(self, reason: str = "manual", symbol: str = None) -> int:
        """Closes open positions on the account.

        Args:
            reason: Human-readable reason for the mass close.
            symbol: If provided, close only positions for this MT5 symbol.
                    If None, close ALL positions on the account.

        Returns:
            int: Number of positions closed successfully.
        """
        loop = asyncio.get_running_loop()

        if symbol:
            positions = await loop.run_in_executor(
                None, lambda: mt5.positions_get(symbol=symbol)
            )
        else:
            positions = await loop.run_in_executor(
                None, lambda: mt5.positions_get()
            )

        if positions is None or len(positions) == 0:
            executor_logger.info(
                f"No open MT5 positions to close"
                f"{f' for {symbol}' if symbol else ''}."
            )
            return 0

        closed = 0
        for pos in positions:
            pos_symbol = pos.symbol
            close_type = mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
            tick = await loop.run_in_executor(
                None, mt5.symbol_info_tick, pos_symbol
            )
            if tick is None:
                executor_logger.error(f"Cannot close ticket {pos.ticket} — no tick data for {pos_symbol}.")
                continue

            close_price = tick.bid if close_type == mt5.ORDER_TYPE_SELL else tick.ask

            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": pos_symbol,
                "volume": pos.volume,
                "type": close_type,
                "position": pos.ticket,
                "price": close_price,
                "deviation": self.deviation,
                "magic": self.magic_number,
                "comment": f"GridMirror_close_{reason}",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }

            result = await loop.run_in_executor(None, lambda r=request: mt5.order_send(r))
            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                side_str = "BUY" if pos.type == mt5.ORDER_TYPE_BUY else "SELL"
                executor_logger.info(
                    f"Closed position — ticket: {pos.ticket} | {side_str} {pos.volume} "
                    f"{pos_symbol} @ {close_price:.2f}"
                )
                closed += 1
            else:
                rc = result.retcode if result else "None"
                executor_logger.error(f"Failed to close ticket {pos.ticket} — retcode: {rc}")

        sym_msg = f" for {symbol}" if symbol else ""
        await self.telegram.send(
            f"🔒 MT5 positions closed{sym_msg} — {closed} position(s) — reason: {reason}"
        )
        executor_logger.info(f"Closed {closed}/{len(positions)} positions{sym_msg} — reason: {reason}")
        return closed

    async def close_pair_positions(self, mt5_symbol: str) -> int:
        """Convenience wrapper to close all positions for a specific MT5 symbol.

        Args:
            mt5_symbol: MT5 symbol name (e.g. "BTCUSD").

        Returns:
            int: Number of positions closed.
        """
        return await self.close_all_positions(reason=f"close_{mt5_symbol}", symbol=mt5_symbol)

    async def get_pair_positions(self, mt5_symbol: str) -> list:
        """Returns open positions for a specific MT5 symbol.

        Args:
            mt5_symbol: MT5 symbol name.

        Returns:
            List of position objects for the symbol.
        """
        loop = asyncio.get_running_loop()
        positions = await loop.run_in_executor(
            None, lambda: mt5.positions_get(symbol=mt5_symbol)
        )
        if positions is None:
            return []
        return list(positions)

    async def get_all_positions_summary(self) -> Dict[str, Any]:
        """Returns a summary of open positions grouped by symbol.

        Returns:
            Dict keyed by MT5 symbol with net_position, unrealised_pnl, open_orders.
        """
        loop = asyncio.get_running_loop()
        positions = await loop.run_in_executor(
            None, lambda: mt5.positions_get()
        )

        summary: Dict[str, Any] = {}
        if positions is None or len(positions) == 0:
            return summary

        for pos in positions:
            sym = pos.symbol
            if sym not in summary:
                summary[sym] = {
                    "net_position": 0.0,
                    "unrealised_pnl": 0.0,
                    "open_orders": 0,
                }
            if pos.type == mt5.ORDER_TYPE_BUY:
                summary[sym]["net_position"] += pos.volume
            else:
                summary[sym]["net_position"] -= pos.volume
            summary[sym]["unrealised_pnl"] += pos.profit
            summary[sym]["open_orders"] += 1

        return summary

    def is_connected(self) -> bool:
        """Returns whether MT5 is currently connected."""
        return self.bot_state.mt5_connected

    async def get_net_position(self) -> float:
        """Returns the net BTC-equivalent position on MT5.

        Buys are positive, sells are negative.
        Updates ``bot_state.mt5_net_position``.

        Returns:
            float: Net position volume.
        """
        loop = asyncio.get_running_loop()
        positions = await loop.run_in_executor(
            None, lambda: mt5.positions_get(symbol=self.symbol)
        )

        if positions is None or len(positions) == 0:
            async with self.bot_state.lock:
                self.bot_state.mt5_net_position = 0.0
            return 0.0

        net = 0.0
        for pos in positions:
            if pos.type == mt5.ORDER_TYPE_BUY:
                net += pos.volume
            else:
                net -= pos.volume

        async with self.bot_state.lock:
            self.bot_state.mt5_net_position = net
        return net

    async def get_account_info(self) -> Dict[str, Any]:
        """Fetches MT5 account info and enforces drawdown kill switches.

        Updates ``bot_state`` equity fields and checks daily-loss /
        max-drawdown thresholds with the configured ``safety_margin``.

        Returns:
            Dict with equity, balance, profit, margin, free_margin.
        """
        loop = asyncio.get_running_loop()
        account = await loop.run_in_executor(None, mt5.account_info)

        if account is None:
            executor_logger.warning("get_account_info — account_info() returned None.")
            return {}

        current_equity = account.equity

        async with self.bot_state.lock:
            self.bot_state.mt5_equity = current_equity
            if current_equity > self.bot_state.mt5_peak_equity:
                self.bot_state.mt5_peak_equity = current_equity

        # ── Drawdown enforcement ───────────────────────────────────
        day_open = self.day_open_equity if self.day_open_equity > 0 else current_equity
        peak = self.bot_state.mt5_peak_equity

        daily_loss_pct = (day_open - current_equity) / day_open if day_open > 0 else 0.0
        max_dd_pct = (peak - current_equity) / peak if peak > 0 else 0.0

        async with self.bot_state.lock:
            self.bot_state.mt5_daily_loss = daily_loss_pct

        # Daily loss check (fire at safety_margin % of limit)
        if daily_loss_pct >= self.daily_loss_limit_pct * self.safety_margin:
            async with self.bot_state.lock:
                if not self.bot_state.kill_switch_daily:
                    self.bot_state.kill_switch_daily = True
                    executor_logger.critical(
                        f"KILL SWITCH: Daily loss {daily_loss_pct:.2%} >= "
                        f"{self.daily_loss_limit_pct * self.safety_margin:.2%} threshold"
                    )
                    await self.telegram.send(
                        f"🚨 KILL SWITCH: Daily loss limit approaching — "
                        f"loss {daily_loss_pct:.2%} / limit {self.daily_loss_limit_pct:.2%}"
                    )

        # Max drawdown check
        if max_dd_pct >= self.max_drawdown_pct * self.safety_margin:
            async with self.bot_state.lock:
                if not self.bot_state.kill_switch_drawdown:
                    self.bot_state.kill_switch_drawdown = True
                    executor_logger.critical(
                        f"KILL SWITCH: Max drawdown {max_dd_pct:.2%} >= "
                        f"{self.max_drawdown_pct * self.safety_margin:.2%} threshold"
                    )
                    await self.telegram.send(
                        f"🚨 KILL SWITCH: Max drawdown limit approaching — "
                        f"drawdown {max_dd_pct:.2%} / limit {self.max_drawdown_pct:.2%}"
                    )

        return {
            "equity": account.equity,
            "balance": account.balance,
            "profit": account.profit,
            "margin": account.margin,
            "free_margin": account.margin_free,
        }

    def reset_day_open_equity(self) -> None:
        """Resets the day-open equity marker to current MT5 equity.

        Called at UTC 00:00 alongside ``bot_state.reset_daily()``.
        """
        self.day_open_equity = self.bot_state.mt5_equity
        executor_logger.info(f"Day-open equity reset to ${self.day_open_equity:,.2f}")
