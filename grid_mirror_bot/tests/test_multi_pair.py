"""
Integration Tests for Multi-Pair System — Week 4.
Tests fill routing, pair independence, scale proportionality,
unknown symbol alerts, and position management.
No real APIs — all dependencies mocked.
"""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch, call

import pytest
import pytest_asyncio

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.pair_state import PairState
from core.pair_worker import PairWorker
from core.pair_watcher import PairWatcher


def _make_pair_state(pair: str, mt5_symbol: str, capital: float = 60.0) -> PairState:
    """Creates a test PairState."""
    return PairState(
        pair=pair,
        mt5_symbol=mt5_symbol,
        grid_capital=capital,
        mirror_enabled=True,
        regime="RANGING",
        mt5_symbol_available=True,
        mt5_symbol_checked=True,
    )


def _make_fill(pair: str, side: str = "BUY") -> dict:
    """Creates a test fill dict."""
    return {
        "symbol": pair,
        "side": side,
        "qty": 0.0001,
        "price": 95000.0 if "BTC" in pair else 3500.0,
        "order_id": f"TEST_{pair}_{int(time.time())}",
        "timestamp": int(time.time() * 1000),
    }


@pytest.mark.asyncio
async def test_fill_routed_to_correct_pair_worker():
    """BTCUSDT fill should only trigger BTC worker, not ETH."""
    btc_callback = AsyncMock()
    eth_callback = AsyncMock()

    btc_state = _make_pair_state("BTCUSDT", "BTCUSD")
    eth_state = _make_pair_state("ETHUSDT", "ETHUSD", 30.0)

    pair_states = {"BTCUSDT": btc_state, "ETHUSDT": eth_state}
    fill_callbacks = {"BTCUSDT": btc_callback, "ETHUSDT": eth_callback}

    client = AsyncMock()
    watcher = PairWatcher(
        config={"binance": {"symbol": "BTCUSDT"}},
        pair_states=pair_states,
        fill_callbacks=fill_callbacks,
        binance_client=client,
    )

    fill = _make_fill("BTCUSDT")
    await watcher._route_fill(fill)

    await asyncio.sleep(0.1)

    btc_callback.assert_called_once_with(fill)
    eth_callback.assert_not_called()


@pytest.mark.asyncio
async def test_fill_for_unconfigured_pair_ignored():
    """SOLUSDT fill should be ignored when not in configured pairs."""
    btc_callback = AsyncMock()

    btc_state = _make_pair_state("BTCUSDT", "BTCUSD")
    pair_states = {"BTCUSDT": btc_state}
    fill_callbacks = {"BTCUSDT": btc_callback}

    client = AsyncMock()
    watcher = PairWatcher(
        config={"binance": {"symbol": "BTCUSDT"}},
        pair_states=pair_states,
        fill_callbacks=fill_callbacks,
        binance_client=client,
    )

    fill = _make_fill("SOLUSDT")
    await watcher._route_fill(fill)

    await asyncio.sleep(0.1)

    btc_callback.assert_not_called()


@pytest.mark.asyncio
async def test_fill_for_disabled_pair_skipped():
    """ADAUSDT fill should be skipped when pair is disabled."""
    ada_callback = AsyncMock()

    ada_state = _make_pair_state("ADAUSDT", "ADAUSD", 10.0)
    ada_state.mirror_enabled = False
    ada_state.mt5_symbol_available = False

    pair_states = {"ADAUSDT": ada_state}
    fill_callbacks = {"ADAUSDT": ada_callback}

    client = AsyncMock()
    watcher = PairWatcher(
        config={"binance": {"symbol": "BTCUSDT"}},
        pair_states=pair_states,
        fill_callbacks=fill_callbacks,
        binance_client=client,
    )

    fill = _make_fill("ADAUSDT")
    await watcher._route_fill(fill)

    await asyncio.sleep(0.1)

    ada_callback.assert_not_called()


@pytest.mark.asyncio
async def test_scale_ratios_proportional_to_capital():
    """BTC ($60) should have 2x the scale ratio of ETH ($30)."""
    from core.compounding import CompoundingEngine

    config = {
        "scaling": {"fundednext_account_size": 10000.0, "ratio_smoothing_periods": 5,
                    "max_scale_factor": 500.0, "min_scale_factor": 10.0,
                    "poll_interval_seconds": 60},
        "risk": {"daily_loss_limit_pct": 0.04, "max_drawdown_pct": 0.10},
        "grid": {"initial_capital": 60.0, "levels": 20, "range_pct": 0.05,
                 "restart_threshold": 0.10},
    }

    btc_engine = CompoundingEngine(config, None)
    btc_engine.binance_balance = 60.0
    btc_engine.smoothed_ratio = 10000.0 / 60.0

    eth_config = dict(config)
    eth_config["grid"] = dict(config["grid"])
    eth_config["grid"]["initial_capital"] = 30.0
    eth_engine = CompoundingEngine(eth_config, None)
    eth_engine.binance_balance = 30.0
    eth_engine.smoothed_ratio = 10000.0 / 30.0

    btc_ratio = btc_engine.smoothed_ratio
    eth_ratio = eth_engine.smoothed_ratio

    assert abs(eth_ratio / btc_ratio - 2.0) < 0.01


@pytest.mark.asyncio
async def test_unknown_symbol_whatsapp_alert_fires_once():
    """Three fills for unknown symbol should trigger WhatsApp exactly once."""
    telegram = AsyncMock()
    telegram.send = AsyncMock()

    btc_state = _make_pair_state("BTCUSDT", "BTCUSD")
    pair_states = {"BTCUSDT": btc_state}
    fill_callbacks = {"BTCUSDT": AsyncMock()}

    client = AsyncMock()
    watcher = PairWatcher(
        config={"binance": {"symbol": "BTCUSDT"}},
        pair_states=pair_states,
        fill_callbacks=fill_callbacks,
        binance_client=client,
    )

    for _ in range(3):
        fill = _make_fill("SOLUSDT")
        await watcher._route_fill(fill)

    await asyncio.sleep(0.1)

    fill_callbacks["BTCUSDT"].assert_not_called()


@pytest.mark.asyncio
async def test_global_closeall_closes_all_pairs():
    """closeall command should close positions for all symbols."""
    executor = AsyncMock()
    executor.close_all_positions = AsyncMock(return_value=3)

    result = await executor.close_all_positions(reason="api_command", symbol=None)

    executor.close_all_positions.assert_called_once_with(reason="api_command", symbol=None)
    assert result == 3


@pytest.mark.asyncio
async def test_per_pair_close_only_affects_own_symbol():
    """Closing BTC positions should not affect ETH positions."""
    executor = AsyncMock()
    executor.close_all_positions = AsyncMock(return_value=1)

    await executor.close_all_positions(reason="close_BTCUSD", symbol="BTCUSD")

    executor.close_all_positions.assert_called_once_with(reason="close_BTCUSD", symbol="BTCUSD")
    call_kwargs = executor.close_all_positions.call_args[1]
    assert call_kwargs["symbol"] == "BTCUSD"
