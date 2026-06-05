"""
Tests for SymbolValidator — Week 4.
All MetaTrader5 calls are mocked.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.symbol_validator import SymbolValidator, ValidationResult


def _make_config() -> dict:
    return {
        "mt5": {"symbol": "BTCUSD"},
        "pairs": {
            "BTCUSDT": {"enabled": True, "mt5_symbol": "BTCUSD", "grid_capital": 60.0},
            "ETHUSDT": {"enabled": True, "mt5_symbol": "ETHUSD", "grid_capital": 30.0},
            "SOLUSDT": {"enabled": True, "mt5_symbol": "SOLUSD", "grid_capital": 10.0},
        },
    }


@patch("core.symbol_validator.mt5")
def test_valid_symbol_returns_ok(mock_mt5):
    """Available symbol should return OK with tick details."""
    mock_mt5.terminal_info.return_value = MagicMock()
    sym_info = MagicMock()
    sym_info.visible = True
    sym_info.trade_tick_size = 0.01
    sym_info.volume_min = 0.01
    sym_info.volume_step = 0.01
    mock_mt5.symbol_info.return_value = sym_info

    tick = MagicMock()
    tick.ask = 95010.0
    tick.bid = 95000.0
    mock_mt5.symbol_info_tick.return_value = tick

    validator = SymbolValidator(_make_config())
    result = validator.validate("BTCUSDT", "BTCUSD")

    assert result.available is True
    assert result.reason == "OK"
    assert result.tick_size == 0.01
    assert result.volume_min == 0.01
    assert result.spread_points == 10.0


@patch("core.symbol_validator.mt5")
def test_unknown_symbol_returns_not_found(mock_mt5):
    """Unknown symbol should return SYMBOL_NOT_FOUND."""
    mock_mt5.terminal_info.return_value = MagicMock()
    mock_mt5.symbol_info.return_value = None

    validator = SymbolValidator(_make_config())
    result = validator.validate("SOLUSDT", "SOLUSD")

    assert result.available is False
    assert result.reason == "SYMBOL_NOT_FOUND"


@patch("core.symbol_validator.mt5")
def test_invisible_symbol_gets_selected_and_validated(mock_mt5):
    """Invisible symbol should be made visible and then validated."""
    mock_mt5.terminal_info.return_value = MagicMock()

    invisible_sym = MagicMock()
    invisible_sym.visible = False

    visible_sym = MagicMock()
    visible_sym.visible = True
    visible_sym.trade_tick_size = 0.001
    visible_sym.volume_min = 0.1
    visible_sym.volume_step = 0.1

    mock_mt5.symbol_info.side_effect = [invisible_sym, visible_sym]
    mock_mt5.symbol_select.return_value = True

    tick = MagicMock()
    tick.ask = 3501.0
    tick.bid = 3500.0
    mock_mt5.symbol_info_tick.return_value = tick

    validator = SymbolValidator(_make_config())
    result = validator.validate("ETHUSDT", "ETHUSD")

    assert result.available is True
    assert result.reason == "OK"
    mock_mt5.symbol_select.assert_called_once_with("ETHUSD", True)


@patch("core.symbol_validator.mt5")
def test_mt5_disconnected_returns_not_connected(mock_mt5):
    """Disconnected MT5 should return MT5_NOT_CONNECTED."""
    mock_mt5.terminal_info.return_value = None

    validator = SymbolValidator(_make_config())
    result = validator.validate("BTCUSDT", "BTCUSD")

    assert result.available is False
    assert result.reason == "MT5_NOT_CONNECTED"


@patch("core.symbol_validator.mt5")
def test_validate_all_pairs_returns_results_for_each(mock_mt5):
    """validate_all_pairs should return a result for every pair."""
    mock_mt5.terminal_info.return_value = MagicMock()

    btc_sym = MagicMock()
    btc_sym.visible = True
    btc_sym.trade_tick_size = 0.01
    btc_sym.volume_min = 0.01
    btc_sym.volume_step = 0.01

    eth_sym = MagicMock()
    eth_sym.visible = True
    eth_sym.trade_tick_size = 0.001
    eth_sym.volume_min = 0.01
    eth_sym.volume_step = 0.01

    def symbol_info_side_effect(symbol):
        if symbol == "BTCUSD":
            return btc_sym
        if symbol == "ETHUSD":
            return eth_sym
        return None

    mock_mt5.symbol_info.side_effect = symbol_info_side_effect

    tick = MagicMock()
    tick.ask = 95010.0
    tick.bid = 95000.0
    mock_mt5.symbol_info_tick.return_value = tick

    config = _make_config()
    pairs = {
        "BTCUSDT": config["pairs"]["BTCUSDT"],
        "ETHUSDT": config["pairs"]["ETHUSDT"],
        "SOLUSDT": config["pairs"]["SOLUSDT"],
    }

    validator = SymbolValidator(config)
    results = validator.validate_all_pairs(pairs)

    assert len(results) == 3
    assert results["BTCUSDT"].available is True
    assert results["ETHUSDT"].available is True
    assert results["SOLUSDT"].available is False
    assert results["SOLUSDT"].reason == "SYMBOL_NOT_FOUND"


@patch("core.symbol_validator.mt5")
def test_get_unknown_pairs_returns_unavailable_only(mock_mt5):
    """get_unknown_pairs should only return pairs where available=False."""
    results = {
        "BTCUSDT": ValidationResult("BTCUSDT", "BTCUSD", True, "OK"),
        "ETHUSDT": ValidationResult("ETHUSDT", "ETHUSD", True, "OK"),
        "SOLUSDT": ValidationResult("SOLUSDT", "SOLUSD", False, "SYMBOL_NOT_FOUND"),
    }

    validator = SymbolValidator(_make_config())
    unknown = validator.get_unknown_pairs(results)

    assert unknown == ["SOLUSDT"]
