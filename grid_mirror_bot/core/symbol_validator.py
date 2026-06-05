"""
Symbol Validator — Week 4 Core Component.
Validates that Binance trading pairs have corresponding MT5 symbols
available and tradeable. Reports tick sizes, volume constraints, and spreads.
"""

import logging
from dataclasses import dataclass
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, List

try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None  # type: ignore[assignment]

# ── Logger setup ────────────────────────────────────────────────────
_logs_dir = Path(__file__).parent.parent / "logs"
_logs_dir.mkdir(parents=True, exist_ok=True)
_log_file = _logs_dir / "symbol_validator.log"

logger = logging.getLogger("symbol_validator")
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    _handler = RotatingFileHandler(_log_file, maxBytes=10 * 1024 * 1024, backupCount=5)
    _formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    _handler.setFormatter(_formatter)
    logger.addHandler(_handler)


@dataclass
class ValidationResult:
    """Result of validating a single MT5 symbol."""

    binance_pair: str
    mt5_symbol: str
    available: bool
    reason: str
    tick_size: float = 0.0
    volume_min: float = 0.0
    volume_step: float = 0.0
    spread_points: float = 0.0


class SymbolValidator:
    """Validates MT5 symbol availability for configured Binance pairs.

    Args:
        config: Full application configuration dictionary.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config

    def validate(self, binance_pair: str, mt5_symbol: str) -> ValidationResult:
        """Validates a single Binance pair against its MT5 symbol.

        Args:
            binance_pair: Binance trading pair (e.g. "BTCUSDT").
            mt5_symbol: Corresponding MT5 symbol (e.g. "BTCUSD").

        Returns:
            ValidationResult with availability status and symbol details.
        """
        if mt5 is None:
            return ValidationResult(
                binance_pair=binance_pair,
                mt5_symbol=mt5_symbol,
                available=False,
                reason="MT5_MODULE_NOT_AVAILABLE",
            )

        terminal_info = mt5.terminal_info()
        if terminal_info is None:
            logger.warning(f"MT5 terminal not connected — cannot validate {mt5_symbol}")
            return ValidationResult(
                binance_pair=binance_pair,
                mt5_symbol=mt5_symbol,
                available=False,
                reason="MT5_NOT_CONNECTED",
            )

        sym_info = mt5.symbol_info(mt5_symbol)
        if sym_info is None:
            logger.warning(f"Symbol {mt5_symbol} not found in MT5 symbol list")
            return ValidationResult(
                binance_pair=binance_pair,
                mt5_symbol=mt5_symbol,
                available=False,
                reason="SYMBOL_NOT_FOUND",
            )

        if not sym_info.visible:
            mt5.symbol_select(mt5_symbol, True)
            sym_info = mt5.symbol_info(mt5_symbol)
            if sym_info is None or not sym_info.visible:
                logger.warning(f"Symbol {mt5_symbol} could not be made visible")
                return ValidationResult(
                    binance_pair=binance_pair,
                    mt5_symbol=mt5_symbol,
                    available=False,
                    reason="SYMBOL_NOT_VISIBLE",
                )

        tick_info = mt5.symbol_info_tick(mt5_symbol)
        spread_points = 0.0
        if tick_info is not None:
            spread_points = tick_info.ask - tick_info.bid

        result = ValidationResult(
            binance_pair=binance_pair,
            mt5_symbol=mt5_symbol,
            available=True,
            reason="OK",
            tick_size=sym_info.trade_tick_size,
            volume_min=sym_info.volume_min,
            volume_step=sym_info.volume_step,
            spread_points=spread_points,
        )

        logger.info(
            f"Validated {binance_pair} -> {mt5_symbol}: OK | "
            f"spread={spread_points:.2f} | min_vol={sym_info.volume_min} | "
            f"vol_step={sym_info.volume_step}"
        )
        return result

    def validate_all_pairs(self, pairs: Dict[str, Dict[str, Any]]) -> Dict[str, ValidationResult]:
        """Validates all configured pairs and logs a summary table.

        Args:
            pairs: Dict of pair configs keyed by Binance pair name.

        Returns:
            Dict of ValidationResult keyed by Binance pair name.
        """
        results: Dict[str, ValidationResult] = {}

        for binance_pair, pair_cfg in pairs.items():
            mt5_symbol = pair_cfg.get("mt5_symbol", "")
            result = self.validate(binance_pair, mt5_symbol)
            results[binance_pair] = result

        self._log_results_table(results)
        return results

    def get_unknown_pairs(self, results: Dict[str, ValidationResult]) -> List[str]:
        """Returns list of pairs where the MT5 symbol is not available.

        Args:
            results: Dict from validate_all_pairs().

        Returns:
            List of Binance pair names that failed validation.
        """
        return [pair for pair, result in results.items() if not result.available]

    def _log_results_table(self, results: Dict[str, ValidationResult]) -> None:
        """Logs a formatted table of validation results."""
        logger.info("=" * 70)
        logger.info(f"{'Pair':<12} {'MT5 Symbol':<12} {'Status':<20} {'Spread':<10} {'Min Vol':<10}")
        logger.info("-" * 70)
        for pair, result in results.items():
            status = result.reason
            spread = f"{result.spread_points:.2f}" if result.available else "N/A"
            vol = f"{result.volume_min}" if result.available else "N/A"
            logger.info(f"{pair:<12} {result.mt5_symbol:<12} {status:<20} {spread:<10} {vol:<10}")
        logger.info("=" * 70)
