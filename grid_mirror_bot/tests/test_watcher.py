"""
Unit tests for the Binance user data WebSocket Watcher.
Uses unittest.mock and pytest-asyncio to mock WebSocket socket connections and API requests.
"""

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from core.watcher import BinanceWatcher


class AsyncContextManagerMock:
    """Mock for async context managers."""

    def __init__(self, mock_obj: Any) -> None:
        self.mock_obj = mock_obj

    async def __aenter__(self) -> Any:
        return self.mock_obj

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        pass


@pytest.mark.asyncio
async def test_fill_callback_called_on_execution_report() -> None:
    """Checks that the user stream processes executionReports with FILLED states."""
    mock_client = AsyncMock()
    mock_client.stream_get_listen_key.return_value = "mock_listen_key"
    mock_compounding = MagicMock()
    mock_db = MagicMock()
    mock_telegram = AsyncMock()
    callback_called = asyncio.Event()
    received_fill = {}

    async def mock_callback(fill: dict) -> None:
        nonlocal received_fill
        received_fill = fill
        callback_called.set()

    watcher = BinanceWatcher(
        config={},
        client=mock_client,
        compounding_engine=mock_compounding,
        db_manager=mock_db,
        telegram_alerter=mock_telegram,
        fill_callback=mock_callback,
    )

    # Mock the socket manager
    mock_socket = AsyncMock()
    # Mock returning one message, then raising a cancelled error to exit the loop
    mock_socket.recv.side_effect = [
        {
            "e": "executionReport",
            "X": "FILLED",
            "s": "BTCUSDT",
            "S": "BUY",
            "q": "0.001",
            "p": "95000.0",
            "i": 12345,
            "T": 999999,
        },
        asyncio.CancelledError(),
    ]

    watcher.bsm.user_socket = MagicMock(
        return_value=AsyncContextManagerMock(mock_socket)
    )

    # Start watcher in the background
    await watcher.start()
    
    # Wait for the callback to be called (with a timeout of 1 second)
    try:
        await asyncio.wait_for(callback_called.wait(), timeout=1.0)
    except asyncio.TimeoutError:
        pass

    await watcher.stop()

    assert callback_called.is_set()
    assert received_fill["symbol"] == "BTCUSDT"
    assert received_fill["side"] == "BUY"
    assert received_fill["qty"] == 0.001
    assert received_fill["price"] == 95000.0
    assert received_fill["order_id"] == "12345"


@pytest.mark.asyncio
async def test_partial_fill_ignored() -> None:
    """Verifies that PARTIALLY_FILLED execution reports are ignored."""
    mock_client = AsyncMock()
    mock_client.stream_get_listen_key.return_value = "mock_listen_key"
    mock_compounding = MagicMock()
    mock_db = MagicMock()
    mock_telegram = AsyncMock()
    callback_called = asyncio.Event()

    async def mock_callback(fill: dict) -> None:
        callback_called.set()

    watcher = BinanceWatcher(
        config={},
        client=mock_client,
        compounding_engine=mock_compounding,
        db_manager=mock_db,
        telegram_alerter=mock_telegram,
        fill_callback=mock_callback,
    )

    mock_socket = AsyncMock()
    mock_socket.recv.side_effect = [
        {
            "e": "executionReport",
            "X": "PARTIALLY_FILLED",
            "s": "BTCUSDT",
            "S": "BUY",
            "q": "0.001",
            "p": "95000.0",
            "i": 12345,
            "T": 999999,
        },
        asyncio.CancelledError(),
    ]

    watcher.bsm.user_socket = MagicMock(
        return_value=AsyncContextManagerMock(mock_socket)
    )

    await watcher.start()
    await asyncio.sleep(0.1)
    await watcher.stop()

    assert not callback_called.is_set()


@pytest.mark.asyncio
async def test_reconnect_logic_uses_exponential_backoff() -> None:
    """Ensures disconnected streams reconnect with exponential backoffs."""
    mock_client = AsyncMock()
    mock_compounding = MagicMock()
    mock_db = MagicMock()
    mock_telegram = AsyncMock()

    watcher = BinanceWatcher(
        config={},
        client=mock_client,
        compounding_engine=mock_compounding,
        db_manager=mock_db,
        telegram_alerter=mock_telegram,
    )

    # Force stream_get_listen_key to raise an exception, testing the backoff
    mock_client.stream_get_listen_key.side_effect = [
        Exception("Conn error 1"),
        Exception("Conn error 2"),
        asyncio.CancelledError(),
    ]

    with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        watcher.should_run = True
        try:
            await watcher._main_loop()
        except asyncio.CancelledError:
            pass

        # Should sleep 2 times: 1st with 2.0s, 2nd with 4.0s
        assert mock_sleep.call_count == 2
        mock_sleep.assert_any_call(2.0)
        mock_sleep.assert_any_call(4.0)


@pytest.mark.asyncio
async def test_listen_key_refresh_scheduled() -> None:
    """Ensures keepalive pings are periodically sent to Binance."""
    mock_client = AsyncMock()
    mock_client.stream_get_listen_key.return_value = "test_key"
    mock_compounding = MagicMock()
    mock_db = MagicMock()
    mock_telegram = AsyncMock()

    watcher = BinanceWatcher(
        config={},
        client=mock_client,
        compounding_engine=mock_compounding,
        db_manager=mock_db,
        telegram_alerter=mock_telegram,
    )
    watcher.listen_key = "test_key"
    watcher.should_run = True

    # Mock asyncio.sleep to exit after one check
    with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        mock_sleep.side_effect = [None, asyncio.CancelledError()]
        try:
            await watcher._keepalive_loop()
        except asyncio.CancelledError:
            pass

        # Check keepalive was pinged once
        mock_client.stream_keepalive.assert_called_once_with("test_key")
        mock_sleep.assert_any_call(1800)
