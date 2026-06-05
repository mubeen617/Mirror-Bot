"""
Compounding and Risk Engine for the Grid Mirror Bot.
Handles scale ratio calculation, EMA smoothing, grid restart monitoring,
and simulated FundedNext MT5 position tracking with daily loss and drawdown checks.
"""

import logging
import threading
import time
from typing import Any, Dict, List

logger = logging.getLogger("compounding_engine")


class CompoundingEngine:
    """Calculates scaling ratios and runs real-time drawdown simulations and safety filters."""

    def __init__(self, config: Dict[str, Any], db_manager: Any = None) -> None:
        """
        Initializes the CompoundingEngine.

        Args:
            config (dict): Configuration loaded from config.yaml and secrets.env.
            db_manager (DatabaseManager, optional): The database manager for logging ratio changes.
        """
        self.config = config
        self.db_manager = db_manager

        # Extract config configurations
        scaling_cfg = config.get("scaling", {})
        risk_cfg = config.get("risk", {})
        grid_cfg = config.get("grid", {})

        self.fundednext_account_size = float(scaling_cfg.get("fundednext_account_size", 10000.0))
        self.min_scale_factor = float(scaling_cfg.get("min_scale_factor", 10.0))
        self.max_scale_factor = float(scaling_cfg.get("max_scale_factor", 500.0))
        self.ratio_smoothing_periods = int(scaling_cfg.get("ratio_smoothing_periods", 5))

        self.daily_loss_limit_pct = float(risk_cfg.get("daily_loss_limit_pct", 0.04))
        self.max_drawdown_pct = float(risk_cfg.get("max_drawdown_pct", 0.10))
        self.restart_threshold = float(grid_cfg.get("restart_threshold", 0.10))

        # Core State
        grid_cap = float(grid_cfg.get("initial_capital", 100.0))
        self.binance_balance = grid_cap
        self.fundednext_equity = self.fundednext_account_size
        self.start_of_day_equity = self.fundednext_account_size
        self.baseline_balance = grid_cap
        self.smoothed_ratio = self.fundednext_account_size / grid_cap if grid_cap > 0.0 else 100.0
        self.smoothed_ratio = max(
            self.min_scale_factor, min(self.smoothed_ratio, self.max_scale_factor)
        )
        self.crash_lockout = False

        # Position tracking (simulated MT5 long/short position)
        self.pos_qty = 0.0
        self.avg_entry_price = 0.0
        self.realized_pnl = 0.0
        self.unrealized_pnl = 0.0
        self.latest_price = 0.0

        # Thread-safety lock for process_fill (called from Flask thread and async watcher)
        self._lock = threading.Lock()

        # Scale ratio history (last 10 entries)
        # Entry format: {"timestamp": float, "raw_ratio": float, "smoothed_ratio": float, "balance": float}
        self.ratio_history: List[Dict[str, Any]] = []

    async def poll(self, client: Any) -> None:
        """
        Queries the Binance account balance, updates the scale ratio,
        and marks the simulated MT5 position to market.

        Args:
            client (AsyncClient): The python-binance client instance.
        """
        try:
            # Query Binance balance
            account_info = await client.get_account()
            usdt_balance = 0.0
            for asset in account_info.get("balances", []):
                if asset["asset"] == "USDT":
                    free = float(asset["free"])
                    locked = float(asset["locked"])
                    usdt_balance = free + locked
                    break

            if usdt_balance <= 0.0:
                logger.warning("Query returned zero or negative balance for USDT.")
                return

            self.binance_balance = usdt_balance

            # If baseline is not set yet, initialize it
            if self.baseline_balance == 0.0:
                self.baseline_balance = usdt_balance

            # Compute raw scale ratio
            raw_ratio = self.fundednext_account_size / usdt_balance

            # Apply EMA smoothing
            # alpha = 2 / (N + 1)
            alpha = 2.0 / (self.ratio_smoothing_periods + 1)
            if self.smoothed_ratio == 0.0:
                self.smoothed_ratio = raw_ratio
            else:
                self.smoothed_ratio = (alpha * raw_ratio) + ((1.0 - alpha) * self.smoothed_ratio)

            # Clamp ratio
            self.smoothed_ratio = max(
                self.min_scale_factor, min(self.smoothed_ratio, self.max_scale_factor)
            )

            # Record in history
            now = time.time()
            self.ratio_history.append(
                {
                    "timestamp": now,
                    "raw_ratio": raw_ratio,
                    "smoothed_ratio": self.smoothed_ratio,
                    "balance": usdt_balance,
                }
            )
            if len(self.ratio_history) > 10:
                self.ratio_history.pop(0)

            # Mark position to market
            if self.pos_qty != 0.0:
                # Query current price
                ticker = await client.get_symbol_ticker(symbol=self.config.get("binance", {}).get("symbol", "BTCUSDT"))
                self.latest_price = float(ticker["price"])
                self.unrealized_pnl = self.pos_qty * (self.latest_price - self.avg_entry_price)
            else:
                self.unrealized_pnl = 0.0

            # Update equity
            self.fundednext_equity = self.start_of_day_equity + self.realized_pnl + self.unrealized_pnl

            # Log to DB
            if self.db_manager:
                await self.db_manager.log_ratio(
                    timestamp=int(now * 1000),
                    binance_balance=self.binance_balance,
                    fn_equity=self.fundednext_equity,
                    raw_ratio=raw_ratio,
                    smoothed_ratio=self.smoothed_ratio,
                )

            logger.info(
                f"Balance: {self.binance_balance:.2f} | smoothed_ratio: {self.smoothed_ratio:.2f} | "
                f"FN Equity: {self.fundednext_equity:.2f} | Pos: {self.pos_qty:.4f} @ ${self.avg_entry_price:.2f}"
            )

        except Exception as e:
            logger.error(f"Error in CompoundingEngine poll loop: {e}", exc_info=True)

    def process_fill(self, fill: Dict[str, Any]) -> Dict[str, Any]:
        """
        Scales a Binance fill to simulated FundedNext dimensions and updates risk metrics.

        Args:
            fill (dict): Fill information with side, symbol, qty, price, order_id, timestamp.

        Returns:
            dict: Scaled order specifications and kill switch state.
        """
        with self._lock:
            binance_qty = float(fill["qty"])
            price = float(fill["price"])
            side = fill["side"]
            transact_time = fill.get("timestamp", int(time.time() * 1000))

            # Check conditions for kill switch
            kill_switch = False
            reason = "OK"

            # Check scale ratio below min
            if self.smoothed_ratio < self.min_scale_factor:
                kill_switch = True
                reason = f"Scale ratio {self.smoothed_ratio:.2f} below min limit {self.min_scale_factor}"

            # Check crash lockout
            elif self.crash_lockout:
                kill_switch = True
                reason = "Crash monitor protective lockout active"

            # Check daily drawdown breach or max drawdown breach
            else:
                daily_loss = self.start_of_day_equity - self.fundednext_equity
                daily_loss_pct = daily_loss / self.start_of_day_equity if self.start_of_day_equity > 0 else 0.0
                max_dd_pct = (self.fundednext_account_size - self.fundednext_equity) / self.fundednext_account_size

                if daily_loss_pct >= self.daily_loss_limit_pct:
                    kill_switch = True
                    reason = f"Daily loss limit reached ({daily_loss_pct:.2%})"
                elif max_dd_pct >= self.max_drawdown_pct:
                    kill_switch = True
                    reason = f"Max drawdown limit reached ({max_dd_pct:.2%})"

            # Compute scaled quantity
            scaled_qty = binance_qty * self.smoothed_ratio

            # Update simulated position and realized P&L if not locked out
            if not kill_switch:
                self._update_position(side, scaled_qty, price)

            # Build output structure
            # Convert Binance symbol e.g., BTCUSDT to MT5 format e.g., BTCUSD
            mt5_symbol = fill["symbol"]
            if mt5_symbol.endswith("USDT"):
                mt5_symbol = mt5_symbol[:-1]  # remove the trailing 'T' for MT5 representation

            return {
                "symbol": mt5_symbol,
                "side": side,
                "binance_qty": binance_qty,
                "scaled_qty": scaled_qty,
                "scale_ratio": self.smoothed_ratio,
                "entry_price": price,
                "kill_switch": kill_switch,
                "reason": reason,
                "timestamp": transact_time,
            }

    def _update_position(self, side: str, scaled_qty: float, price: float) -> None:
        """
        Updates the internal simulated long/short position and calculates realized P&L.
        """
        if side.upper() == "BUY":
            if self.pos_qty >= 0.0:
                # Add to long position
                new_qty = self.pos_qty + scaled_qty
                self.avg_entry_price = (
                    ((self.pos_qty * self.avg_entry_price) + (scaled_qty * price)) / new_qty
                    if new_qty > 0
                    else 0.0
                )
                self.pos_qty = new_qty
            else:
                # Close short position
                short_qty = abs(self.pos_qty)
                if short_qty >= scaled_qty:
                    self.realized_pnl += scaled_qty * (self.avg_entry_price - price)
                    self.pos_qty += scaled_qty
                else:
                    self.realized_pnl += short_qty * (self.avg_entry_price - price)
                    self.pos_qty = scaled_qty - short_qty
                    self.avg_entry_price = price
        else:  # SELL
            if self.pos_qty <= 0.0:
                # Add to short position
                short_qty = abs(self.pos_qty)
                new_qty = short_qty + scaled_qty
                self.avg_entry_price = (
                    ((short_qty * self.avg_entry_price) + (scaled_qty * price)) / new_qty
                    if new_qty > 0
                    else 0.0
                )
                self.pos_qty = -new_qty
            else:
                # Close long position
                long_qty = self.pos_qty
                if long_qty >= scaled_qty:
                    self.realized_pnl += scaled_qty * (price - self.avg_entry_price)
                    self.pos_qty -= scaled_qty
                else:
                    self.realized_pnl += long_qty * (price - self.avg_entry_price)
                    self.pos_qty = -(scaled_qty - long_qty)
                    self.avg_entry_price = price

        self.latest_price = price
        self.unrealized_pnl = self.pos_qty * (self.latest_price - self.avg_entry_price) if self.pos_qty != 0.0 else 0.0
        self.fundednext_equity = self.start_of_day_equity + self.realized_pnl + self.unrealized_pnl

    def check_grid_restart(self) -> bool:
        """
        Returns True when current balance has shifted significantly from baseline.
        """
        if self.baseline_balance == 0.0:
            return False
        pct_change = abs(self.binance_balance - self.baseline_balance) / self.baseline_balance
        return pct_change >= self.restart_threshold

    def confirm_restart(self, new_balance: float) -> None:
        """
        Resets baseline balance following a grid restart.

        Args:
            new_balance (float): New starting baseline balance.
        """
        self.baseline_balance = new_balance
        logger.info(f"Baseline balance reset to {new_balance:.2f} following grid restart.")
    def reset_daily(self) -> None:
        """Resets the daily simulated metrics in the compounding engine."""
        self.start_of_day_equity = self.fundednext_equity
        self.realized_pnl = 0.0
        self.unrealized_pnl = 0.0
        logger.info(
            f"Compounding engine daily stats reset — start_of_day_equity set to "
            f"${self.start_of_day_equity:,.2f}"
        )

    def _get_usdt_rate(self) -> float:
        """Returns the USDT/USD conversion rate from config."""
        multi_pair_cfg = self.config.get("multi_pair", {})
        rate_setting = multi_pair_cfg.get("usdt_usd_rate", "auto")
        fallback = float(multi_pair_cfg.get("usdt_fallback_rate", 1.0))

        if rate_setting == "auto":
            return fallback
        try:
            return float(rate_setting)
        except (TypeError, ValueError):
            return fallback

    def _apply_usdt_conversion(self, usdt_amount: float) -> float:
        """Converts a USDT amount to USD using the configured rate."""
        return usdt_amount * self._get_usdt_rate()

    def get_status(self) -> Dict[str, Any]:
        """
        Returns full compounding engine state.
        """
        daily_loss = self.start_of_day_equity - self.fundednext_equity
        daily_loss_pct = daily_loss / self.start_of_day_equity if self.start_of_day_equity > 0 else 0.0
        max_dd_pct = (self.fundednext_account_size - self.fundednext_equity) / self.fundednext_account_size

        return {
            "binance_balance": self.binance_balance,
            "baseline_balance": self.baseline_balance,
            "fn_equity": self.fundednext_equity,
            "start_of_day_equity": self.start_of_day_equity,
            "scale_ratio": self.smoothed_ratio,
            "daily_loss_pct": daily_loss_pct,
            "daily_loss_limit_pct": self.daily_loss_limit_pct,
            "drawdown_pct": max_dd_pct,
            "max_drawdown_limit_pct": self.max_drawdown_pct,
            "pos_qty": self.pos_qty,
            "avg_entry_price": self.avg_entry_price,
            "realized_pnl": self.realized_pnl,
            "unrealized_pnl": self.unrealized_pnl,
            "kill_switch": (
                self.smoothed_ratio < self.min_scale_factor
                or self.crash_lockout
                or daily_loss_pct >= self.daily_loss_limit_pct
                or max_dd_pct >= self.max_drawdown_pct
            ),
        }
