"""
Unit tests for the Crash Monitor — Week 3.

All Binance REST calls are fully mocked.  Tests verify:
    - No trigger below threshold
    - Trigger fires at threshold and executes crash protocol in order
    - No double-trigger (lockout)
    - Weekend threshold is higher
    - Old prices are dropped from window
    - Reset clears lockout
    - Crash protocol step order
    - Insufficient window data — no trigger
"""

import asyncio
import collections
import time
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch, call, PropertyMock

import pytest

from core.bot_state import BotState
from core.crash_monitor import CrashMonitor


# ── Helpers ─────────────────────────────────────────────────────────

def _get_test_config(overrides: dict | None = None) -> dict:
    """Returns a config dict with standard crash monitor thresholds."""
    cfg = {
        "crash_monitor": {
            "drop_threshold_pct": 0.015,
            "window_seconds": 300,
            "poll_interval_seconds": 5,
            "weekend_drop_threshold": 0.020,
            "require_manual_restart": True,
        },
    }
    if overrides:
        cfg["crash_monitor"].update(overrides)
    return cfg


def _make_monitor(
    config: dict | None = None,
    bot_state: BotState | None = None,
    executor: AsyncMock | None = None,
    telegram: AsyncMock | None = None,
    binance_client: AsyncMock | None = None,
) -> CrashMonitor:
    """Creates a CrashMonitor with sensible test defaults."""
    return CrashMonitor(
        config=config or _get_test_config(),
        bot_state=bot_state or BotState(),
        executor=executor or AsyncMock(),
        telegram=telegram or AsyncMock(),
        binance_client=binance_client or AsyncMock(),
    )


def _seed_window(
    monitor: CrashMonitor,
    oldest_price: float,
    count: int = 60,
    window_span_seconds: int = 250,
) -> None:
    """Pre-fills the price window with ``count`` entries at ``oldest_price``.

    Entries span from ``window_span_seconds`` ago to ~recently, so they
    stay within the 300 s window.
    """
    now_ms = int(time.time() * 1000)
    interval_ms = (window_span_seconds * 1000) // count
    for i in range(count):
        ts = now_ms - (count - i) * interval_ms
        monitor._price_window.append((ts, oldest_price))


def _mock_weekday():
    """Context manager to mock datetime.now() to return a Wednesday (weekday=2)."""
    return patch("core.crash_monitor.datetime", wraps=datetime, **{
        "now.return_value": MagicMock(weekday=MagicMock(return_value=2))
    })


def _mock_weekend():
    """Context manager to mock datetime.now() to return a Saturday (weekday=5)."""
    return patch("core.crash_monitor.datetime", wraps=datetime, **{
        "now.return_value": MagicMock(weekday=MagicMock(return_value=5))
    })


# ── Tests ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_no_trigger_below_threshold() -> None:
    """Drop = 1.4% (below 1.5%) → crash protocol NOT fired."""
    state = BotState()
    executor = AsyncMock()
    client = AsyncMock()
    # Current price 98.6 → (100 - 98.6) / 100 = 1.4%
    client.get_symbol_ticker.return_value = {"price": "98.6"}

    monitor = _make_monitor(
        bot_state=state, executor=executor, binance_client=client
    )
    _seed_window(monitor, oldest_price=100.0, count=20)

    with _mock_weekday():
        await monitor._check()

    executor.close_all_positions.assert_not_called()
    assert state.crash_lockout is False
    assert state.regime != "CRASH"


@pytest.mark.asyncio
async def test_trigger_fires_at_threshold() -> None:
    """Drop = 1.6% (above 1.5%) → full crash protocol fires."""
    state = BotState()
    executor = AsyncMock()
    executor.close_all_positions.return_value = 2
    tg = AsyncMock()
    client = AsyncMock()
    # Current price 98.4 → (100 - 98.4) / 100 = 1.6%
    client.get_symbol_ticker.return_value = {"price": "98.4"}

    monitor = _make_monitor(
        bot_state=state, executor=executor, telegram=tg, binance_client=client
    )
    _seed_window(monitor, oldest_price=100.0, count=20)

    with _mock_weekday():
        await monitor._check()

    executor.close_all_positions.assert_called_once()
    assert state.crash_lockout is True
    assert state.regime == "CRASH"
    assert state.mirror_enabled is False
    tg.send.assert_called_once()
    assert "CRASH PROTOCOL" in tg.send.call_args[0][0]


@pytest.mark.asyncio
async def test_no_double_trigger() -> None:
    """After first trigger, second check at same drop → no second trigger."""
    state = BotState()
    executor = AsyncMock()
    executor.close_all_positions.return_value = 1
    client = AsyncMock()
    client.get_symbol_ticker.return_value = {"price": "98.0"}

    monitor = _make_monitor(
        bot_state=state, executor=executor, binance_client=client
    )
    _seed_window(monitor, oldest_price=100.0, count=20)

    with _mock_weekday():
        # First check — fires
        await monitor._check()
        assert executor.close_all_positions.call_count == 1

        # Second check — should NOT fire again
        await monitor._check()
        assert executor.close_all_positions.call_count == 1


@pytest.mark.asyncio
async def test_weekend_threshold_higher() -> None:
    """On Saturday, drop = 1.6% (above weekday 1.5% but below weekend 2.0%) → no trigger."""
    state = BotState()
    executor = AsyncMock()
    client = AsyncMock()
    # 1.6% drop — above weekday threshold but below weekend
    client.get_symbol_ticker.return_value = {"price": "98.4"}

    monitor = _make_monitor(
        bot_state=state, executor=executor, binance_client=client
    )
    _seed_window(monitor, oldest_price=100.0, count=20)

    with _mock_weekend():
        await monitor._check()

    executor.close_all_positions.assert_not_called()
    assert state.crash_lockout is False


@pytest.mark.asyncio
async def test_old_prices_dropped_from_window() -> None:
    """Prices older than 300s are evicted from the window."""
    state = BotState()
    executor = AsyncMock()
    client = AsyncMock()
    client.get_symbol_ticker.return_value = {"price": "100.0"}

    monitor = _make_monitor(
        bot_state=state, executor=executor, binance_client=client
    )

    # Add an old entry (400 seconds ago) at a much higher price
    now_ms = int(time.time() * 1000)
    old_ts = now_ms - 400_000  # 400 seconds ago
    monitor._price_window.append((old_ts, 200.0))

    # Add 20 recent entries at 100
    _seed_window(monitor, oldest_price=100.0, count=20, window_span_seconds=100)

    with _mock_weekday():
        await monitor._check()

    # The old 200.0 entry should have been evicted
    # So drop should be ~0% (all entries around 100)
    executor.close_all_positions.assert_not_called()


@pytest.mark.asyncio
async def test_reset_clears_lockout() -> None:
    """After crash trigger, reset() clears lockout flags."""
    state = BotState()
    executor = AsyncMock()
    executor.close_all_positions.return_value = 1
    client = AsyncMock()
    client.get_symbol_ticker.return_value = {"price": "98.0"}

    monitor = _make_monitor(
        bot_state=state, executor=executor, binance_client=client
    )
    _seed_window(monitor, oldest_price=100.0, count=20)

    # Trigger crash (use weekday threshold)
    with _mock_weekday():
        await monitor._check()
    assert state.crash_lockout is True
    assert monitor._triggered is True

    # Reset
    monitor.reset()
    assert state.crash_lockout is False
    assert monitor._triggered is False


@pytest.mark.asyncio
async def test_crash_protocol_step_order() -> None:
    """Verifies positions are closed BEFORE crash_lockout is set.

    Uses mock call-order tracking to ensure step ordering.
    """
    state = BotState()
    call_order: list[str] = []

    async def mock_close(**kwargs):
        call_order.append("close_positions")
        return 1

    executor = AsyncMock()
    executor.close_all_positions.side_effect = mock_close

    tg = AsyncMock()
    client = AsyncMock()
    client.get_symbol_ticker.return_value = {"price": "98.0"}

    monitor = _make_monitor(
        bot_state=state, executor=executor, telegram=tg, binance_client=client
    )
    _seed_window(monitor, oldest_price=100.0, count=20)

    # Monkey-patch bot_state property to track when lockout is set
    original_setattr = state.__class__.__setattr__

    def tracking_setattr(self, name, value):
        if name == "crash_lockout" and value is True:
            call_order.append("set_lockout")
        original_setattr(self, name, value)

    with _mock_weekday():
        with patch.object(type(state), "__setattr__", tracking_setattr):
            await monitor._check()

    assert call_order.index("close_positions") < call_order.index("set_lockout"), (
        f"Expected close_positions before set_lockout, got: {call_order}"
    )


@pytest.mark.asyncio
async def test_insufficient_window_data_no_trigger() -> None:
    """Only 5 price readings in window (<10 minimum) → no trigger."""
    state = BotState()
    executor = AsyncMock()
    client = AsyncMock()
    # Even a huge drop should not trigger with insufficient data
    client.get_symbol_ticker.return_value = {"price": "50.0"}

    monitor = _make_monitor(
        bot_state=state, executor=executor, binance_client=client
    )

    # Seed only 4 entries
    now_ms = int(time.time() * 1000)
    for i in range(4):
        monitor._price_window.append((now_ms - (4 - i) * 5000, 100.0))

    with _mock_weekday():
        await monitor._check()

    # Now window has 5 entries (4 old + 1 new from _check) → still < 10
    executor.close_all_positions.assert_not_called()
    assert state.crash_lockout is False
