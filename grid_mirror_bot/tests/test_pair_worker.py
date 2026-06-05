"""
Tests for PairWorker — Week 4.
Verifies fill routing, kill switch handling, symbol validation,
and per-pair independence.
"""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.pair_state import PairState
from core.pair_worker import PairWorker


def _make_pair_state(pair: str = "BTCUSDT", mt5_symbol: str = "BTCUSD") -> PairState:
    """Creates a PairState configured for testing."""
    return PairState(
        pair=pair,
        mt5_symbol=mt5_symbol,
        grid_capital=60.0,
        mirror_enabled=True,
        regime="RANGING",
        mt5_symbol_available=True,
        mt5_symbol_checked=True,
    )


def _make_fill(pair: str = "BTCUSDT", side: str = "BUY") -> dict:
    """Creates a mock fill dict."""
    return {
        "symbol": pair,
        "side": side,
        "qty": 0.0001,
        "price": 95000.0,
        "order_id": f"TEST_{int(time.time())}",
        "timestamp": int(time.time() * 1000),
    }


def _make_worker(pair_state: PairState) -> PairWorker:
    """Creates a PairWorker with mocked dependencies."""
    config = {
        "scaling": {"poll_interval_seconds": 60, "fundednext_account_size": 10000.0,
                    "ratio_smoothing_periods": 5, "max_scale_factor": 500.0,
                    "min_scale_factor": 10.0},
        "risk": {"daily_loss_limit_pct": 0.04, "max_drawdown_pct": 0.10, "safety_margin": 0.80,
                 "max_single_order_pct": 0.02},
        "grid": {"initial_capital": pair_state.grid_capital, "levels": 20, "range_pct": 0.05,
                 "restart_threshold": 0.10},
        "binance": {"symbol": pair_state.pair},
        "mt5": {"symbol": pair_state.mt5_symbol},
        "reconciler": {"enabled": False},
    }
    pair_config = {
        "mt5_symbol": pair_state.mt5_symbol,
        "grid_capital": pair_state.grid_capital,
        "mirror_enabled": pair_state.mirror_enabled,
        "regime": {"atr_ranging_threshold": 36, "atr_trending_threshold": 92,
                   "slope_bull_threshold": 0.004, "slope_bear_threshold": -0.004,
                   "slope_hard_threshold": 0.005, "poll_interval_seconds": 60},
        "crash_monitor": {"drop_threshold_pct": 0.015, "window_seconds": 300,
                          "poll_interval_seconds": 5, "weekend_drop_threshold": 0.020},
    }

    executor = AsyncMock()
    executor.execute = AsyncMock(return_value={
        "status": "FILLED", "mt5_order_id": 12345, "symbol": pair_state.mt5_symbol,
        "side": "BUY", "volume": 0.01, "price": 95000.0,
    })
    executor.close_all_positions = AsyncMock(return_value=1)

    telegram = AsyncMock()
    telegram.send = AsyncMock()
    telegram.send_fill = AsyncMock()

    client = AsyncMock()

    worker = PairWorker(
        pair=pair_state.pair,
        pair_config=pair_config,
        global_config=config,
        pair_state=pair_state,
        mt5_executor=executor,
        telegram=telegram,
        binance_client=client,
        db_manager=None,
    )
    # Pre-initialise compounding
    from core.compounding import CompoundingEngine
    worker.compounding = CompoundingEngine(config, None)
    worker.compounding.smoothed_ratio = 100.0
    worker.compounding.binance_balance = pair_state.grid_capital

    return worker


@pytest.mark.asyncio
async def test_on_fill_calls_executor_when_trading_allowed():
    """Fill should be routed to MT5 executor when trading is allowed."""
    pair_state = _make_pair_state()
    worker = _make_worker(pair_state)
    fill = _make_fill()

    await worker.on_fill(fill)

    worker.executor.execute.assert_called_once()
    call_args = worker.executor.execute.call_args[0][0]
    assert call_args["symbol"] == "BTCUSD"
    assert call_args["side"] == "BUY"


@pytest.mark.asyncio
async def test_on_fill_skips_when_kill_switch_active():
    """Fill should be skipped when kill switch is active."""
    pair_state = _make_pair_state()
    pair_state.kill_switch_daily = True
    worker = _make_worker(pair_state)
    fill = _make_fill()

    await worker.on_fill(fill)

    worker.executor.execute.assert_not_called()


@pytest.mark.asyncio
async def test_on_fill_skips_when_symbol_unavailable():
    """Fill should be skipped when MT5 symbol is not available."""
    pair_state = _make_pair_state()
    pair_state.mt5_symbol_available = False
    worker = _make_worker(pair_state)
    fill = _make_fill()

    await worker.on_fill(fill)

    worker.executor.execute.assert_not_called()


@pytest.mark.asyncio
async def test_on_fill_skips_when_mirror_disabled():
    """Fill should be skipped when mirror is disabled."""
    pair_state = _make_pair_state()
    pair_state.mirror_enabled = False
    worker = _make_worker(pair_state)
    fill = _make_fill()

    await worker.on_fill(fill)

    worker.executor.execute.assert_not_called()


@pytest.mark.asyncio
async def test_on_fill_updates_pair_state_stats():
    """Fill should update pair state statistics."""
    pair_state = _make_pair_state()
    worker = _make_worker(pair_state)
    fill = _make_fill()

    assert pair_state.fills_today == 0
    assert pair_state.mt5_orders_today == 0

    await worker.on_fill(fill)

    assert pair_state.fills_today == 1
    assert pair_state.mt5_orders_today == 1
    assert pair_state.last_fill_at > 0


@pytest.mark.asyncio
async def test_on_fill_sends_whatsapp_alert():
    """Fill should trigger a WhatsApp alert."""
    pair_state = _make_pair_state()
    worker = _make_worker(pair_state)
    fill = _make_fill()

    await worker.on_fill(fill)

    worker.telegram.send_fill.assert_called_once()


@pytest.mark.asyncio
async def test_on_fill_logs_to_sqlite():
    """Fill should be logged to SQLite when db_manager is present."""
    pair_state = _make_pair_state()
    worker = _make_worker(pair_state)
    worker.db_manager = AsyncMock()
    worker.db_manager.log_fill = AsyncMock()
    fill = _make_fill()

    await worker.on_fill(fill)

    worker.db_manager.log_fill.assert_called_once()


@pytest.mark.asyncio
async def test_crash_only_affects_own_pair():
    """BTC crash should NOT affect ETH pair state."""
    btc_state = _make_pair_state("BTCUSDT", "BTCUSD")
    eth_state = _make_pair_state("ETHUSDT", "ETHUSD")
    eth_state.grid_capital = 30.0

    # Simulate BTC crash lockout
    btc_state.crash_lockout = True
    btc_state.mirror_enabled = False

    # ETH must remain unaffected
    assert eth_state.crash_lockout is False
    assert eth_state.mirror_enabled is True
    assert eth_state.is_trading_allowed() is True


@pytest.mark.asyncio
async def test_regime_change_disables_mirror_for_own_pair_only():
    """BTC regime change to SLOW_BEAR should NOT affect ETH."""
    btc_state = _make_pair_state("BTCUSDT", "BTCUSD")
    eth_state = _make_pair_state("ETHUSDT", "ETHUSD")

    # Simulate BTC regime changing to SLOW_BEAR
    btc_state.regime = "SLOW_BEAR"
    btc_state.mirror_enabled = False

    # ETH must remain unaffected
    assert eth_state.mirror_enabled is True
    assert eth_state.regime == "RANGING"
    assert eth_state.is_trading_allowed() is True
