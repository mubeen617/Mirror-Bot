"""
Unit tests for the Compounding and Risk Engine.
Uses pytest-asyncio and mock clients to validate scaling, smoothing, restarts, and drawdown limits.
"""

import pytest
from core.compounding import CompoundingEngine


class MockClient:
    """Mock Binance REST client for tests."""

    def __init__(self, balance: float = 100.0, price: float = 100000.0) -> None:
        self.balance = balance
        self.price = price

    async def get_account(self) -> dict:
        """Mock account balance query."""
        free = self.balance / 2.0
        locked = self.balance / 2.0
        return {
            "balances": [
                {"asset": "USDT", "free": str(free), "locked": str(locked)}
            ]
        }

    async def get_symbol_ticker(self, symbol: str) -> dict:
        """Mock symbol pricing query."""
        return {"price": str(self.price)}


def get_test_config() -> dict:
    """Helper configuration for unit tests."""
    return {
        "binance": {
            "symbol": "BTCUSDT",
        },
        "scaling": {
            "fundednext_account_size": 10000.0,
            "min_scale_factor": 10.0,
            "max_scale_factor": 500.0,
            "ratio_smoothing_periods": 5,
        },
        "risk": {
            "daily_loss_limit_pct": 0.04,
            "max_drawdown_pct": 0.10,
        },
        "grid": {
            "restart_threshold": 0.10,
        },
    }


@pytest.mark.asyncio
async def test_scale_ratio_calculation() -> None:
    """Validates simple scale ratio math."""
    config = get_test_config()
    engine = CompoundingEngine(config)
    client = MockClient(balance=100.0)

    await engine.poll(client)
    assert engine.binance_balance == 100.0
    # First poll: smoothed = raw = 10000 / 100 = 100.0
    assert engine.smoothed_ratio == 100.0


@pytest.mark.asyncio
async def test_ema_smoothing() -> None:
    """Validates that EMA smoothing converges over multiple polls."""
    config = get_test_config()
    engine = CompoundingEngine(config)
    
    # 1st Poll: balance = 100.0 (raw = 100.0) -> smoothed = 100.0
    client = MockClient(balance=100.0)
    await engine.poll(client)
    assert engine.smoothed_ratio == 100.0

    # 2nd Poll: balance = 50.0 (raw = 200.0) -> alpha = 2 / (5 + 1) = 1/3
    # smoothed = (1/3 * 200) + (2/3 * 100) = 133.33
    client.balance = 50.0
    await engine.poll(client)
    assert pytest.approx(engine.smoothed_ratio, 0.01) == 133.33


@pytest.mark.asyncio
async def test_scale_ratio_clamping() -> None:
    """Validates min and max ratio boundaries."""
    config = get_test_config()
    engine = CompoundingEngine(config)

    # Trigger max clamp: raw = 10000/10 = 1000. EMA brings it up over multiple polls.
    # After enough polls the EMA converges and hits the 500.0 max clamp.
    client = MockClient(balance=10.0)
    for _ in range(20):
        await engine.poll(client)
    assert engine.smoothed_ratio == 500.0

    # Trigger min clamp (10000 / 2000.0 = 5.0 -> min 10.0)
    # After enough polls the EMA converges below min and gets clamped to 10.0
    client.balance = 2000.0
    for _ in range(20):
        await engine.poll(client)
    assert engine.smoothed_ratio == 10.0


@pytest.mark.asyncio
async def test_process_fill_returns_correct_scaled_qty() -> None:
    """Validates quantity scaling and formatting."""
    config = get_test_config()
    engine = CompoundingEngine(config)
    client = MockClient(balance=100.0)
    await engine.poll(client)

    fill = {
        "symbol": "BTCUSDT",
        "side": "BUY",
        "qty": 0.001,
        "price": 95000.0,
        "order_id": "123",
        "timestamp": 1234567890123,
    }
    scaled = engine.process_fill(fill)
    
    assert scaled["symbol"] == "BTCUSD"
    assert scaled["side"] == "BUY"
    assert scaled["binance_qty"] == 0.001
    assert scaled["scaled_qty"] == 0.001 * 100.0
    assert scaled["scale_ratio"] == 100.0
    assert scaled["entry_price"] == 95000.0
    assert scaled["kill_switch"] is False
    assert scaled["reason"] == "OK"


@pytest.mark.asyncio
async def test_kill_switch_fires_on_drawdown_breach() -> None:
    """Checks that the kill switch is activated if daily loss limits are broken."""
    config = get_test_config()
    engine = CompoundingEngine(config)
    client = MockClient(balance=100.0)
    await engine.poll(client)

    # Buy position
    fill1 = {"symbol": "BTCUSDT", "side": "BUY", "qty": 0.5, "price": 100000.0, "order_id": "1", "timestamp": 123}
    # Scaled qty = 0.5 * 100.0 = 50.0. Avg price = 100000
    engine.process_fill(fill1)

    # Mark to market drop to $90k (10% drop on 50 units = $500,000 loss!)
    # Max loss is 4% of $10,000 = $400. This will easily breach drawdown.
    client.price = 90000.0
    await engine.poll(client)

    fill2 = {"symbol": "BTCUSDT", "side": "BUY", "qty": 0.001, "price": 90000.0, "order_id": "2", "timestamp": 124}
    scaled = engine.process_fill(fill2)
    assert scaled["kill_switch"] is True
    assert "loss limit reached" in scaled["reason"].lower()


@pytest.mark.asyncio
async def test_kill_switch_fires_on_crash_lockout() -> None:
    """Checks that the kill switch triggers if a protective crash lockout is active."""
    config = get_test_config()
    engine = CompoundingEngine(config)
    engine.smoothed_ratio = 100.0
    engine.crash_lockout = True

    fill = {"symbol": "BTCUSDT", "side": "BUY", "qty": 0.001, "price": 95000.0, "order_id": "1", "timestamp": 123}
    scaled = engine.process_fill(fill)
    assert scaled["kill_switch"] is True
    assert "crash monitor" in scaled["reason"].lower()


@pytest.mark.asyncio
async def test_grid_restart_threshold_triggers() -> None:
    """Verifies that grid restart triggers when threshold shifts by 10%."""
    config = get_test_config()
    engine = CompoundingEngine(config)
    client = MockClient(balance=100.0)

    await engine.poll(client)
    assert engine.check_grid_restart() is False

    # Shift balance by 11% (threshold 10%)
    client.balance = 111.0
    await engine.poll(client)
    assert engine.check_grid_restart() is True


@pytest.mark.asyncio
async def test_confirm_restart_resets_baseline() -> None:
    """Validates that a restart confirmation establishes a new baseline."""
    config = get_test_config()
    engine = CompoundingEngine(config)
    client = MockClient(balance=100.0)
    await engine.poll(client)

    client.balance = 111.0
    await engine.poll(client)
    assert engine.check_grid_restart() is True

    # Resets baseline
    engine.confirm_restart(111.0)
    assert engine.check_grid_restart() is False


@pytest.mark.asyncio
async def test_get_status_returns_all_required_keys() -> None:
    """Asserts key structures in engine status summaries."""
    config = get_test_config()
    engine = CompoundingEngine(config)
    status = engine.get_status()

    required_keys = {
        "binance_balance",
        "baseline_balance",
        "fn_equity",
        "start_of_day_equity",
        "scale_ratio",
        "daily_loss_pct",
        "daily_loss_limit_pct",
        "drawdown_pct",
        "max_drawdown_limit_pct",
        "pos_qty",
        "avg_entry_price",
        "realized_pnl",
        "unrealized_pnl",
        "kill_switch",
    }
    for key in required_keys:
        assert key in status
