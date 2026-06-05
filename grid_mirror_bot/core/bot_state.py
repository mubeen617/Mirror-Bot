"""
Shared Bot State — Single Source of Truth.
Thread-safe dataclass used by all components via dependency injection.
All writes are protected by asyncio.Lock to prevent data races.
"""

import asyncio
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict


@dataclass
class BotState:
    """Centralised state container shared across all bot components.

    Fields are grouped by subsystem. All mutations must go through
    the ``update()`` context manager or ``reset_daily()`` to guarantee
    thread-safety under the shared ``asyncio.Lock``.
    """

    # ── Regime ──────────────────────────────────────────────────────
    regime: str = "RANGING"
    atr: float = 0.0
    slope_pct: float = 0.0
    band_24h: float = 0.0

    # ── Crash monitor ──────────────────────────────────────────────
    crash_lockout: bool = False
    mirror_enabled: bool = True
    btc_price: float = 0.0
    drop_5m_pct: float = 0.0

    # ── Binance ────────────────────────────────────────────────────
    binance_balance: float = 0.0
    binance_net_position: float = 0.0   # net BTC held

    # ── FundedNext MT5 ─────────────────────────────────────────────
    mt5_connected: bool = False
    mt5_equity: float = 0.0
    mt5_net_position: float = 0.0       # net BTC-equivalent position
    mt5_daily_loss: float = 0.0
    mt5_peak_equity: float = 0.0

    # ── Scaling ────────────────────────────────────────────────────
    scale_ratio: float = 0.0
    smoothed_ratio: float = 0.0

    # ── Kill switches ──────────────────────────────────────────────
    kill_switch_daily: bool = False
    kill_switch_drawdown: bool = False

    # ── Grid ───────────────────────────────────────────────────────
    grid_restart_due: bool = False
    grid_baseline: float = 100.0

    # ── Session stats ──────────────────────────────────────────────
    fills_today: int = 0
    mt5_orders_today: int = 0
    last_fill_timestamp: int = 0
    last_reconcile_timestamp: int = 0
    session_start: float = 0.0

    # ── Drift tracking ─────────────────────────────────────────────
    position_drift: float = 0.0         # Binance net - (MT5 net / scale_ratio)
    drift_corrections_today: int = 0

    # ── Internal (excluded from dataclass __init__ default) ────────
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False, compare=False)

    # ────────────────────────────────────────────────────────────────
    # Public API
    # ────────────────────────────────────────────────────────────────

    @property
    def lock(self) -> asyncio.Lock:
        """Public access to the state lock for external components."""
        return self._lock

    def to_dict(self) -> Dict[str, Any]:
        """Serialises every public field to a plain dict for JSON responses.

        Returns:
            Dict[str, Any]: All state values keyed by field name.
        """
        data = asdict(self)
        # Remove the private lock field (not serialisable)
        data.pop("_lock", None)
        return data

    async def reset_daily(self) -> None:
        """Resets daily counters — called at UTC 00:00.

        Counters reset:
        - fills_today
        - mt5_orders_today
        - kill_switch_daily
        - drift_corrections_today
        - mt5_daily_loss
        """
        async with self._lock:
            self.fills_today = 0
            self.mt5_orders_today = 0
            self.kill_switch_daily = False
            self.drift_corrections_today = 0
            self.mt5_daily_loss = 0.0

    def is_trading_allowed(self) -> bool:
        """Checks all safety conditions to determine if trading is permitted.

        Returns:
            False if *any* of the following are true:
            - ``crash_lockout``
            - ``kill_switch_daily``
            - ``kill_switch_drawdown``
            - ``mirror_enabled`` is False
            - ``mt5_connected`` is False
        """
        if self.crash_lockout:
            return False
        if self.kill_switch_daily:
            return False
        if self.kill_switch_drawdown:
            return False
        if not self.mirror_enabled:
            return False
        if not self.mt5_connected:
            return False
        return True

    def trading_blocked_reason(self) -> str:
        """Returns a human-readable reason why trading is blocked.

        Returns:
            str: Reason string, or ``'OK'`` if trading is allowed.
        """
        if self.crash_lockout:
            return "CRASH_LOCKOUT"
        if self.kill_switch_daily:
            return "DAILY_LOSS_LIMIT"
        if self.kill_switch_drawdown:
            return "MAX_DRAWDOWN_LIMIT"
        if not self.mirror_enabled:
            return "MIRROR_DISABLED"
        if not self.mt5_connected:
            return "MT5_DISCONNECTED"
        return "OK"
