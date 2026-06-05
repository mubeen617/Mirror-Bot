"""
Unit tests for the Position Reconciler.
The executor, Binance client, and Telegram alerter are fully mocked.
"""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.bot_state import BotState
from core.reconciler import Reconciler


# ── Helpers ─────────────────────────────────────────────────────────

def _get_test_config() -> dict:
    """Returns a minimal config dict for reconciler tests."""
    return {
        "reconciler": {
            "interval_seconds": 60,
            "drift_threshold_btc": 0.001,
            "max_corrections_per_day": 10,
        },
        "binance": {
            "symbol": "BTCUSDT",
        },
        "mt5": {
            "symbol": "BTCUSD",
        },
    }


def _mock_binance_account(btc_balance: float = 0.0) -> dict:
    """Returns a mock Binance get_account() response."""
    return {
        "balances": [
            {"asset": "BTC", "free": str(btc_balance / 2), "locked": str(btc_balance / 2)},
            {"asset": "USDT", "free": "1000.0", "locked": "0.0"},
        ]
    }


def _make_reconciler(
    bot_state: BotState | None = None,
    mt5_net: float = 0.0,
    btc_balance: float = 0.0,
) -> tuple:
    """Creates a Reconciler with mocked dependencies.

    Returns:
        (reconciler, executor_mock, binance_mock, telegram_mock)
    """
    state = bot_state or BotState()

    executor = AsyncMock()
    executor.get_net_position = AsyncMock(return_value=mt5_net)
    executor.execute = AsyncMock(return_value={"status": "FILLED", "mt5_order_id": 99999})
    executor.symbol = "BTCUSD"

    binance_client = AsyncMock()
    binance_client.get_account = AsyncMock(return_value=_mock_binance_account(btc_balance))

    telegram = AsyncMock()

    recon = Reconciler(
        config=_get_test_config(),
        bot_state=state,
        executor=executor,
        binance_client=binance_client,
        telegram=telegram,
    )
    return recon, executor, binance_client, telegram


# ── Tests ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_no_correction_when_drift_below_threshold() -> None:
    """Drift within threshold (< 0.001) should NOT fire a correction order.

    Setup:
        binance_net = 0.001 BTC, smoothed_ratio = 100
        expected_mt5 = 0.001 * 100 = 0.100
        mt5_net = 0.099
        drift = 0.100 - 0.099 = 0.001 → exactly at threshold → within (< check)
    """
    state = BotState(smoothed_ratio=100.0, mt5_connected=True, mirror_enabled=True)
    recon, executor, _, _ = _make_reconciler(
        bot_state=state,
        mt5_net=0.0999,  # drift = 0.100 - 0.0999 = 0.0001 < 0.001
        btc_balance=0.001,
    )

    await recon._reconcile()

    executor.execute.assert_not_called()


@pytest.mark.asyncio
async def test_correction_fired_when_drift_above_threshold() -> None:
    """Drift above threshold should fire a BUY correction.

    Setup:
        binance_net = 0.002 BTC, smoothed_ratio = 100
        expected_mt5 = 0.002 * 100 = 0.200
        mt5_net = 0.050
        drift = 0.200 - 0.050 = 0.150 → well above 0.001 threshold
    """
    state = BotState(smoothed_ratio=100.0, mt5_connected=True, mirror_enabled=True)
    recon, executor, _, _ = _make_reconciler(
        bot_state=state,
        mt5_net=0.050,
        btc_balance=0.002,
    )

    await recon._reconcile()

    executor.execute.assert_called_once()
    order_arg = executor.execute.call_args[0][0]
    assert order_arg["side"] == "BUY"
    assert pytest.approx(order_arg["scaled_qty"], abs=0.001) == 0.150


@pytest.mark.asyncio
async def test_sell_correction_when_mt5_over_positioned() -> None:
    """When MT5 has MORE position than expected, fire a SELL correction.

    Setup:
        binance_net = 0.001, smoothed_ratio = 100
        expected_mt5 = 0.001 * 100 = 0.100
        mt5_net = 0.200
        drift = 0.100 - 0.200 = -0.100 → SELL correction needed
    """
    state = BotState(smoothed_ratio=100.0, mt5_connected=True, mirror_enabled=True)
    recon, executor, _, _ = _make_reconciler(
        bot_state=state,
        mt5_net=0.200,
        btc_balance=0.001,
    )

    await recon._reconcile()

    executor.execute.assert_called_once()
    order_arg = executor.execute.call_args[0][0]
    assert order_arg["side"] == "SELL"
    assert pytest.approx(order_arg["scaled_qty"], abs=0.001) == 0.100


@pytest.mark.asyncio
async def test_no_correction_when_trading_not_allowed() -> None:
    """Even with large drift, no correction should fire if trading is blocked.

    Sets crash_lockout = True to block trading.
    """
    state = BotState(
        smoothed_ratio=100.0,
        mt5_connected=True,
        mirror_enabled=True,
        crash_lockout=True,  # blocks trading
    )
    recon, executor, _, telegram = _make_reconciler(
        bot_state=state,
        mt5_net=0.0,
        btc_balance=0.01,  # expected = 1.0, drift = 1.0 → huge
    )

    await recon._reconcile()

    executor.execute.assert_not_called()
    # Telegram should still alert about the drift
    telegram.send.assert_called()


@pytest.mark.asyncio
async def test_max_corrections_per_day_enforced() -> None:
    """When max daily corrections reached, no more corrections should fire."""
    state = BotState(
        smoothed_ratio=100.0,
        mt5_connected=True,
        mirror_enabled=True,
        drift_corrections_today=10,  # max = 10 in config
    )
    recon, executor, _, telegram = _make_reconciler(
        bot_state=state,
        mt5_net=0.0,
        btc_balance=0.01,  # large drift
    )

    await recon._reconcile()

    executor.execute.assert_not_called()
    # Should send CRITICAL alert
    calls = [str(c) for c in telegram.send.call_args_list]
    critical_sent = any("CRITICAL" in c for c in calls)
    assert critical_sent, "Expected CRITICAL alert about max corrections"


@pytest.mark.asyncio
async def test_reconcile_updates_last_reconcile_timestamp() -> None:
    """After reconciliation, last_reconcile_timestamp should be updated."""
    state = BotState(smoothed_ratio=100.0, mt5_connected=True, mirror_enabled=True)
    recon, _, _, _ = _make_reconciler(
        bot_state=state,
        mt5_net=0.100,  # drift within threshold
        btc_balance=0.001,
    )

    before = state.last_reconcile_timestamp
    await recon._reconcile()

    assert state.last_reconcile_timestamp > before
    assert state.last_reconcile_timestamp > 0
