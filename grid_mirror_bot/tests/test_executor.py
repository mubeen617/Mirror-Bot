"""
Unit tests for the MT5 Executor.
MetaTrader5 is fully mocked — no real MT5 connection is used in tests.
"""

import asyncio
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.bot_state import BotState
from core.executor import MT5Executor


# ── Helpers ─────────────────────────────────────────────────────────

def _get_test_config() -> dict:
    """Returns a minimal config dict for executor tests."""
    return {
        "mt5": {
            "login": 12345678,
            "password": "testpass",
            "server": "TestServer",
            "symbol": "BTCUSD",
            "deviation": 20,
            "magic_number": 20260001,
            "demo_mode": True,
            "connection_timeout_seconds": 30,
            "reconnect_interval_seconds": 10,
            "max_reconnect_attempts": 10,
        },
        "risk": {
            "daily_loss_limit_pct": 0.04,
            "max_drawdown_pct": 0.10,
            "safety_margin": 0.80,
        },
    }


def _make_executor(
    bot_state: BotState | None = None,
    telegram: AsyncMock | None = None,
) -> MT5Executor:
    """Creates an MT5Executor with sensible test defaults."""
    state = bot_state or BotState()
    tg = telegram or AsyncMock()
    return MT5Executor(_get_test_config(), state, tg)


def _mock_symbol_info() -> SimpleNamespace:
    """Returns a mock mt5.symbol_info() result."""
    return SimpleNamespace(
        visible=True,
        volume_step=0.01,
        volume_min=0.01,
    )


def _mock_tick(ask: float = 95000.0, bid: float = 94990.0) -> SimpleNamespace:
    """Returns a mock mt5.symbol_info_tick() result."""
    return SimpleNamespace(ask=ask, bid=bid)


def _mock_order_result(retcode: int, order_id: int = 12345, price: float = 95000.0) -> SimpleNamespace:
    """Returns a mock mt5.order_send() result."""
    return SimpleNamespace(
        retcode=retcode,
        order=order_id,
        price=price,
        comment="OK" if retcode == 10009 else "Rejected",
    )


def _base_scaled_order(side: str = "BUY", qty: float = 0.05) -> dict:
    """Returns a base scaled order dict for tests."""
    return {
        "symbol": "BTCUSD",
        "side": side,
        "scaled_qty": qty,
        "scale_ratio": 100.0,
        "entry_price": 95000.0,
        "kill_switch": False,
        "reason": "OK",
        "timestamp": int(time.time() * 1000),
        "order_id": "88721",
    }


# ── Tests ───────────────────────────────────────────────────────────

# TRADE_RETCODE_DONE = 10009 in real MT5
RETCODE_DONE = 10009
RETCODE_REJECT = 10006


@pytest.mark.asyncio
@patch("core.executor.mt5")
async def test_execute_places_buy_order_correctly(mock_mt5: MagicMock) -> None:
    """Validates that a BUY order is built and sent with correct parameters."""
    mock_mt5.TRADE_ACTION_DEAL = 1
    mock_mt5.ORDER_TYPE_BUY = 0
    mock_mt5.ORDER_TYPE_SELL = 1
    mock_mt5.ORDER_TIME_GTC = 0
    mock_mt5.ORDER_FILLING_IOC = 1
    mock_mt5.TRADE_RETCODE_DONE = RETCODE_DONE

    mock_mt5.symbol_info.return_value = _mock_symbol_info()
    mock_mt5.symbol_info_tick.return_value = _mock_tick()
    mock_mt5.order_send.return_value = _mock_order_result(RETCODE_DONE, order_id=99999, price=95000.0)

    state = BotState(mt5_connected=True, mirror_enabled=True)
    executor = _make_executor(bot_state=state)

    result = await executor.execute(_base_scaled_order("BUY", 0.05))

    assert result["status"] == "FILLED"
    assert result["mt5_order_id"] == 99999
    assert result["side"] == "BUY"
    assert result["symbol"] == "BTCUSD"
    assert result["volume"] == 0.05

    # Verify the request sent to order_send
    call_args = mock_mt5.order_send.call_args[0][0]
    assert call_args["symbol"] == "BTCUSD"
    assert call_args["volume"] == 0.05
    assert call_args["type"] == 0  # ORDER_TYPE_BUY
    assert call_args["magic"] == 20260001


@pytest.mark.asyncio
@patch("core.executor.mt5")
async def test_execute_places_sell_order_correctly(mock_mt5: MagicMock) -> None:
    """Validates that a SELL order is built and sent with correct parameters."""
    mock_mt5.TRADE_ACTION_DEAL = 1
    mock_mt5.ORDER_TYPE_BUY = 0
    mock_mt5.ORDER_TYPE_SELL = 1
    mock_mt5.ORDER_TIME_GTC = 0
    mock_mt5.ORDER_FILLING_IOC = 1
    mock_mt5.TRADE_RETCODE_DONE = RETCODE_DONE

    mock_mt5.symbol_info.return_value = _mock_symbol_info()
    mock_mt5.symbol_info_tick.return_value = _mock_tick(ask=95100.0, bid=95090.0)
    mock_mt5.order_send.return_value = _mock_order_result(RETCODE_DONE, order_id=10001, price=95090.0)

    state = BotState(mt5_connected=True, mirror_enabled=True)
    executor = _make_executor(bot_state=state)

    result = await executor.execute(_base_scaled_order("SELL", 0.03))

    assert result["status"] == "FILLED"
    assert result["side"] == "SELL"
    assert result["volume"] == 0.03

    call_args = mock_mt5.order_send.call_args[0][0]
    assert call_args["type"] == 1  # ORDER_TYPE_SELL
    assert call_args["price"] == 95090.0  # bid for SELL


@pytest.mark.asyncio
@patch("core.executor.mt5")
async def test_execute_skipped_when_kill_switch_active(mock_mt5: MagicMock) -> None:
    """Verifies execute returns SKIPPED when daily kill switch is active."""
    state = BotState(
        mt5_connected=True,
        mirror_enabled=True,
        kill_switch_daily=True,
    )
    executor = _make_executor(bot_state=state)

    result = await executor.execute(_base_scaled_order())

    assert result["status"] == "SKIPPED"
    assert result["reason"] == "DAILY_LOSS_LIMIT"
    mock_mt5.order_send.assert_not_called()


@pytest.mark.asyncio
@patch("core.executor.mt5")
async def test_execute_skipped_when_mt5_disconnected(mock_mt5: MagicMock) -> None:
    """Verifies execute returns SKIPPED when MT5 is disconnected."""
    state = BotState(mt5_connected=False, mirror_enabled=True)
    executor = _make_executor(bot_state=state)

    result = await executor.execute(_base_scaled_order())

    assert result["status"] == "SKIPPED"
    assert result["reason"] == "MT5_DISCONNECTED"
    mock_mt5.order_send.assert_not_called()


@pytest.mark.asyncio
@patch("core.executor.mt5")
async def test_execute_returns_failed_on_bad_retcode(mock_mt5: MagicMock) -> None:
    """Verifies execute returns FAILED when MT5 rejects the order."""
    mock_mt5.TRADE_ACTION_DEAL = 1
    mock_mt5.ORDER_TYPE_BUY = 0
    mock_mt5.ORDER_TYPE_SELL = 1
    mock_mt5.ORDER_TIME_GTC = 0
    mock_mt5.ORDER_FILLING_IOC = 1
    mock_mt5.TRADE_RETCODE_DONE = RETCODE_DONE

    mock_mt5.symbol_info.return_value = _mock_symbol_info()
    mock_mt5.symbol_info_tick.return_value = _mock_tick()
    mock_mt5.order_send.return_value = _mock_order_result(RETCODE_REJECT)

    state = BotState(mt5_connected=True, mirror_enabled=True)
    executor = _make_executor(bot_state=state)

    result = await executor.execute(_base_scaled_order())

    assert result["status"] == "FAILED"
    assert result["retcode"] == RETCODE_REJECT


@pytest.mark.asyncio
@patch("core.executor.mt5")
async def test_close_all_positions_closes_each_position(mock_mt5: MagicMock) -> None:
    """Verifies that close_all_positions sends a closing order for each open position."""
    mock_mt5.ORDER_TYPE_BUY = 0
    mock_mt5.ORDER_TYPE_SELL = 1
    mock_mt5.TRADE_ACTION_DEAL = 1
    mock_mt5.ORDER_TIME_GTC = 0
    mock_mt5.ORDER_FILLING_IOC = 1
    mock_mt5.TRADE_RETCODE_DONE = RETCODE_DONE

    # Mock 3 open positions
    pos1 = SimpleNamespace(ticket=1001, type=0, volume=0.01, symbol="BTCUSD")  # BUY
    pos2 = SimpleNamespace(ticket=1002, type=0, volume=0.02, symbol="BTCUSD")  # BUY
    pos3 = SimpleNamespace(ticket=1003, type=1, volume=0.01, symbol="BTCUSD")  # SELL
    mock_mt5.positions_get.return_value = [pos1, pos2, pos3]
    mock_mt5.symbol_info_tick.return_value = _mock_tick()
    mock_mt5.order_send.return_value = _mock_order_result(RETCODE_DONE)

    state = BotState(mt5_connected=True, mirror_enabled=True)
    tg = AsyncMock()
    executor = _make_executor(bot_state=state, telegram=tg)

    count = await executor.close_all_positions(reason="test")

    assert count == 3
    assert mock_mt5.order_send.call_count == 3
    tg.send.assert_called()  # Telegram notification sent


@pytest.mark.asyncio
@patch("core.executor.mt5")
async def test_close_all_positions_returns_zero_when_no_positions(mock_mt5: MagicMock) -> None:
    """Verifies close_all_positions returns 0 when there are no open positions."""
    mock_mt5.positions_get.return_value = None

    state = BotState(mt5_connected=True, mirror_enabled=True)
    executor = _make_executor(bot_state=state)

    count = await executor.close_all_positions()

    assert count == 0


@pytest.mark.asyncio
@patch("core.executor.mt5")
async def test_drawdown_kill_switch_fires_at_80_pct_of_limit(mock_mt5: MagicMock) -> None:
    """Validates kill switch fires when daily loss >= 80% of the 4% limit (i.e. >= 3.2%).

    - At 96.1% of day_open (daily loss = 3.9%): below the 4% limit itself
      but >= 80% of 4% = 3.2% → kill switch should fire.
    - First we check a loss below the safety threshold does NOT fire.
    """
    state = BotState(mt5_connected=True, mirror_enabled=True)
    tg = AsyncMock()
    executor = _make_executor(bot_state=state, telegram=tg)
    executor.day_open_equity = 10000.0

    # Scenario 1: Equity = 96.9% of day_open → daily loss = 3.1%
    # 3.1% < (4% * 80% = 3.2%) → should NOT trigger
    account_ok = SimpleNamespace(equity=9690.0, balance=9690.0, profit=-310.0, margin=0.0, margin_free=9690.0)
    mock_mt5.account_info.return_value = account_ok
    state.mt5_peak_equity = 10000.0

    await executor.get_account_info()
    assert state.kill_switch_daily is False

    # Scenario 2: Equity = 95.9% of day_open → daily loss = 4.1%
    # 4.1% >= (4% * 80% = 3.2%) → SHOULD trigger
    account_bad = SimpleNamespace(equity=9590.0, balance=9590.0, profit=-410.0, margin=0.0, margin_free=9590.0)
    mock_mt5.account_info.return_value = account_bad

    await executor.get_account_info()
    assert state.kill_switch_daily is True
    tg.send.assert_called()


@pytest.mark.asyncio
@patch("core.executor.mt5")
async def test_get_net_position_sums_correctly(mock_mt5: MagicMock) -> None:
    """Validates net position = sum(buy volumes) - sum(sell volumes)."""
    mock_mt5.ORDER_TYPE_BUY = 0
    mock_mt5.ORDER_TYPE_SELL = 1

    pos1 = SimpleNamespace(type=0, volume=0.01)  # BUY +0.01
    pos2 = SimpleNamespace(type=0, volume=0.01)  # BUY +0.01
    pos3 = SimpleNamespace(type=1, volume=0.005)  # SELL -0.005
    mock_mt5.positions_get.return_value = [pos1, pos2, pos3]

    state = BotState(mt5_connected=True, mirror_enabled=True)
    executor = _make_executor(bot_state=state)

    net = await executor.get_net_position()

    assert pytest.approx(net, abs=1e-8) == 0.015
    assert pytest.approx(state.mt5_net_position, abs=1e-8) == 0.015
