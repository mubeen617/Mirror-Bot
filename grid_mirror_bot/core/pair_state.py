"""
Per-Pair State — Week 4 Core Component.
Thread-safe dataclass storing all runtime state for a single trading pair.
One instance per active pair. All writes protected by asyncio.Lock.
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class PairState:
    """Centralised state container for a single trading pair.

    All mutations must be performed under the ``lock`` to guarantee
    thread-safety across concurrent async tasks.
    """

    # ── Identity ───────────────────────────────────────────────────
    pair: str = ""
    mt5_symbol: str = ""
    grid_capital: float = 0.0

    # ── Mirror control ─────────────────────────────────────────────
    mirror_enabled: bool = True
    crash_lockout: bool = False

    # ── Regime ─────────────────────────────────────────────────────
    regime: str = "STARTING"
    prev_regime: str = ""
    atr: float = 0.0
    slope_pct: float = 0.0
    band_24h: float = 0.0
    regime_updated_at: float = 0.0

    # ── Price ──────────────────────────────────────────────────────
    current_price: float = 0.0
    drop_5m_pct: float = 0.0
    price_updated_at: float = 0.0

    # ── Binance position ───────────────────────────────────────────
    binance_net_position: float = 0.0

    # ── MT5 position ───────────────────────────────────────────────
    mt5_net_position: float = 0.0
    mt5_equity_allocated: float = 0.0

    # ── Scaling ────────────────────────────────────────────────────
    scale_ratio: float = 0.0
    smoothed_ratio: float = 0.0

    # ── Kill switches ──────────────────────────────────────────────
    kill_switch_daily: bool = False
    kill_switch_drawdown: bool = False

    # ── Session stats ──────────────────────────────────────────────
    fills_today: int = 0
    mt5_orders_today: int = 0
    last_fill_at: float = 0.0
    last_reconcile_at: float = 0.0
    drift_corrections_today: int = 0
    position_drift: float = 0.0

    # ── Symbol availability ────────────────────────────────────────
    mt5_symbol_available: bool = True
    mt5_symbol_checked: bool = False

    # ── Component health ───────────────────────────────────────────
    watcher_alive: bool = False
    watcher_last_seen: float = 0.0
    regime_detector_alive: bool = False
    regime_detector_last_seen: float = 0.0
    crash_monitor_alive: bool = False
    crash_monitor_last_seen: float = 0.0
    reconciler_alive: bool = False
    reconciler_last_seen: float = 0.0

    # ── Internal ───────────────────────────────────────────────────
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False, compare=False)

    @property
    def lock(self) -> asyncio.Lock:
        """Public access to the state lock for external components."""
        return self._lock

    def is_trading_allowed(self) -> bool:
        """Checks all safety conditions to determine if trading is permitted for this pair.

        Returns:
            False if any blocking condition is active.
        """
        return (
            self.mirror_enabled
            and not self.crash_lockout
            and not self.kill_switch_daily
            and not self.kill_switch_drawdown
            and self.mt5_symbol_available
            and self.regime in ("RANGING", "SLOW_BULL")
        )

    def trading_blocked_reason(self) -> str:
        """Returns a human-readable reason why trading is blocked for this pair."""
        if not self.mirror_enabled:
            return "MIRROR_DISABLED"
        if self.crash_lockout:
            return "CRASH_LOCKOUT"
        if self.kill_switch_daily:
            return "DAILY_LOSS_LIMIT"
        if self.kill_switch_drawdown:
            return "MAX_DRAWDOWN_LIMIT"
        if not self.mt5_symbol_available:
            return "MT5_SYMBOL_UNAVAILABLE"
        if self.regime not in ("RANGING", "SLOW_BULL"):
            return f"REGIME_{self.regime}"
        return "OK"

    def to_dict(self) -> Dict[str, Any]:
        """Serialises all public fields to a plain dict for JSON responses."""
        return {
            "pair": self.pair,
            "mt5_symbol": self.mt5_symbol,
            "grid_capital": self.grid_capital,
            "mirror_enabled": self.mirror_enabled,
            "crash_lockout": self.crash_lockout,
            "regime": self.regime,
            "prev_regime": self.prev_regime,
            "atr": self.atr,
            "slope_pct": self.slope_pct,
            "band_24h": self.band_24h,
            "regime_updated_at": self.regime_updated_at,
            "current_price": self.current_price,
            "drop_5m_pct": self.drop_5m_pct,
            "price_updated_at": self.price_updated_at,
            "binance_net_position": self.binance_net_position,
            "mt5_net_position": self.mt5_net_position,
            "mt5_equity_allocated": self.mt5_equity_allocated,
            "scale_ratio": self.scale_ratio,
            "smoothed_ratio": self.smoothed_ratio,
            "kill_switch_daily": self.kill_switch_daily,
            "kill_switch_drawdown": self.kill_switch_drawdown,
            "fills_today": self.fills_today,
            "mt5_orders_today": self.mt5_orders_today,
            "last_fill_at": self.last_fill_at,
            "last_reconcile_at": self.last_reconcile_at,
            "drift_corrections_today": self.drift_corrections_today,
            "position_drift": self.position_drift,
            "mt5_symbol_available": self.mt5_symbol_available,
            "mt5_symbol_checked": self.mt5_symbol_checked,
            "watcher_alive": self.watcher_alive,
            "regime_detector_alive": self.regime_detector_alive,
            "crash_monitor_alive": self.crash_monitor_alive,
            "reconciler_alive": self.reconciler_alive,
            "is_trading_allowed": self.is_trading_allowed(),
            "trading_blocked_reason": self.trading_blocked_reason(),
        }

    async def reset_daily(self) -> None:
        """Resets daily counters — called at UTC 00:00."""
        async with self._lock:
            self.fills_today = 0
            self.mt5_orders_today = 0
            self.kill_switch_daily = False
            self.drift_corrections_today = 0
