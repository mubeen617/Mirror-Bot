"""
Regime Detector — Week 3 Core Component.

Classifies the BTC market into one of four regimes by analysing
1 440 one-minute candles fetched from the Binance REST API every
polling interval:

    RANGING        – low ATR, flat slope → ideal grid conditions
    SLOW_BULL      – moderate uptrend     → shift grid range upward
    SLOW_BEAR      – moderate downtrend   → pause bots
    TRENDING_HARD  – high ATR / steep slope / wide band → pause bots

On every regime change the detector:
    1. Logs at WARNING level
    2. Sends a Telegram alert with old → new regime and recommended action
    3. Calls ``_on_regime_change`` which can pause / resume the mirror bot
"""

import asyncio
import logging
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.bot_state import BotState

# ── Logger setup ────────────────────────────────────────────────────
_logs_dir = Path(__file__).parent.parent / "logs"
_logs_dir.mkdir(parents=True, exist_ok=True)
_log_file = _logs_dir / "regime_detector.log"

logger = logging.getLogger("regime_detector")
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    _handler = RotatingFileHandler(_log_file, maxBytes=10 * 1024 * 1024, backupCount=5)
    _formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    _handler.setFormatter(_formatter)
    logger.addHandler(_handler)

# Regime → recommended action mapping
REGIME_ACTIONS: Dict[str, str] = {
    "RANGING": "Bots RUNNING — ideal grid conditions",
    "SLOW_BULL": "Bots RUNNING — shift grid range upward every 3-4%",
    "SLOW_BEAR": "Bots PAUSED — waiting for recovery",
    "TRENDING_HARD": "Bots PAUSED — high volatility detected",
}


class RegimeDetector:
    """Periodically classifies the BTC market regime and updates shared state.

    Args:
        config: Full application configuration dictionary.
        bot_state: Shared ``BotState`` instance.
        telegram: ``TelegramAlerter`` for regime-change notifications.
        binance_client: ``binance.AsyncClient`` for REST kline queries.
        executor: Optional ``MT5Executor`` for closing positions on pause regimes.
        db_manager: Optional ``DatabaseManager`` for persisting regime logs.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        bot_state: BotState,
        telegram: Any,
        binance_client: Any,
        executor: Optional[Any] = None,
        db_manager: Optional[Any] = None,
    ) -> None:
        self.config = config
        self.bot_state = bot_state
        self.telegram = telegram
        self.client = binance_client
        self.executor = executor
        self.db_manager = db_manager

        regime_cfg = config.get("regime", {})
        self._atr_ranging = float(regime_cfg.get("atr_ranging_threshold", 36))
        self._atr_trending = float(regime_cfg.get("atr_trending_threshold", 92))
        self._slope_bull = float(regime_cfg.get("slope_bull_threshold", 0.004))
        self._slope_bear = float(regime_cfg.get("slope_bear_threshold", -0.004))
        self._slope_hard = float(regime_cfg.get("slope_hard_threshold", 0.005))
        self._poll_interval = int(regime_cfg.get("poll_interval_seconds", 60))

        self._prev_regime: str = "INITIAL"

    # ────────────────────────────────────────────────────────────────
    # Public API
    # ────────────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Runs the regime classification loop forever.

        Each iteration sleeps ``poll_interval_seconds`` then calls
        ``_classify()``.  Exceptions are logged at ERROR and never
        propagate — the loop always continues.
        """
        logger.info(
            "Regime detector started — polling every %d seconds", self._poll_interval
        )
        while True:
            try:
                await self._classify()
            except asyncio.CancelledError:
                logger.info("Regime detector cancelled.")
                raise
            except Exception as exc:
                logger.error("Error in regime detector: %s", exc, exc_info=True)
            await asyncio.sleep(self._poll_interval)

    # ────────────────────────────────────────────────────────────────
    # Classification
    # ────────────────────────────────────────────────────────────────

    async def _classify(self) -> None:
        """Fetches candles, computes indicators, classifies regime, and acts."""
        # 1. Fetch last 1 440 one-minute candles
        klines = await self.client.get_klines(
            symbol="BTCUSDT", interval="1m", limit=1440
        )
        if not klines or len(klines) < 15:
            logger.warning("Insufficient kline data — received %d candles", len(klines) if klines else 0)
            return

        # 2. Extract price arrays
        highs: List[float] = [float(k[2]) for k in klines]
        lows: List[float] = [float(k[3]) for k in klines]
        closes: List[float] = [float(k[4]) for k in klines]

        # 3. Compute ATR-14
        atr = self._compute_atr(highs, lows, closes, period=14)
        self.bot_state.atr = atr

        # 4. Compute slope (last 60 closes — 1 hour)
        slope_pct = self._compute_slope(closes, window=60)
        self.bot_state.slope_pct = slope_pct

        # 5. Compute 24h band
        band = self._compute_band(highs, lows)
        self.bot_state.band_24h = band

        # 6. Classify regime
        regime = self._determine_regime(atr, slope_pct, band)

        logger.debug(
            "Classification — regime=%s | ATR=%.2f | slope=%.4f%%/h | band=%.4f",
            regime, atr, slope_pct, band,
        )

        # 7. If regime changed
        if regime != self._prev_regime and self._prev_regime != "INITIAL":
            logger.warning(
                "Regime change: %s → %s | ATR=%.2f | slope=%.4f%%/h | band=%.4f",
                self._prev_regime, regime, atr, slope_pct, band,
            )
            action = REGIME_ACTIONS.get(regime, "Unknown")
            await self.telegram.send(
                f"📊 Regime: {self._prev_regime} → {regime}\n{action}"
            )
            await self._on_regime_change(self._prev_regime, regime)

        # 8. Always — update state
        prev_for_log = self._prev_regime
        self.bot_state.regime = regime
        self._prev_regime = regime

        # Update BTC price from last close
        self.bot_state.btc_price = closes[-1]

        # Persist to SQLite
        if self.db_manager is not None:
            try:
                await self.db_manager.log_regime(
                    timestamp=int(time.time()),
                    regime=regime,
                    atr=atr,
                    slope_pct=slope_pct,
                    band_24h=band,
                    prev_regime=prev_for_log,
                )
            except Exception as db_err:
                logger.error("Failed to write regime_log: %s", db_err)

    # ────────────────────────────────────────────────────────────────
    # Regime change handler
    # ────────────────────────────────────────────────────────────────

    async def _on_regime_change(self, prev: str, new: str) -> None:
        """Executes side-effects when the regime transitions.

        - Pause regimes (SLOW_BEAR, TRENDING_HARD):
              Disable mirror, close FundedNext positions.
        - Run regimes (RANGING, SLOW_BULL):
              Notify user — never auto-restart.

        Args:
            prev: Previous regime label.
            new: New regime label.
        """
        if new in ("SLOW_BEAR", "TRENDING_HARD"):
            self.bot_state.mirror_enabled = False
            if self.executor is not None:
                try:
                    await self.executor.close_all_positions(reason=f"regime_{new}")
                except Exception as exc:
                    logger.error("Failed to close positions on regime change: %s", exc)
            await self.telegram.send(
                f"Regime: {prev} → {new}\n"
                f"{REGIME_ACTIONS.get(new, '')}\n"
                f"FundedNext positions closed."
            )
        elif new in ("RANGING", "SLOW_BULL"):
            # NEVER auto-restart — always manual
            await self.telegram.send(
                f"Regime: {prev} → {new}\n"
                f"Market recovering. Send /start command to resume bots."
            )

    # ────────────────────────────────────────────────────────────────
    # Indicator calculations
    # ────────────────────────────────────────────────────────────────

    @staticmethod
    def _compute_atr(
        highs: List[float],
        lows: List[float],
        closes: List[float],
        period: int = 14,
    ) -> float:
        """Computes the Average True Range over ``period`` candles.

        True Range for candle *i* =
            max(high-low, |high-prev_close|, |low-prev_close|)

        Args:
            highs: High prices.
            lows: Low prices.
            closes: Close prices.
            period: Number of TR values to average.

        Returns:
            ATR value as float.
        """
        tr_list: List[float] = []
        for i in range(1, len(highs)):
            tr1 = highs[i] - lows[i]
            tr2 = abs(highs[i] - closes[i - 1])
            tr3 = abs(lows[i] - closes[i - 1])
            tr_list.append(max(tr1, tr2, tr3))

        if not tr_list:
            return 0.0
        recent = tr_list[-period:]
        return sum(recent) / len(recent)

    @staticmethod
    def _compute_slope(closes: List[float], window: int = 60) -> float:
        """Linear regression slope on the last ``window`` closes, as % per hour.

        slope_per_minute = (n·Σ(xy) − Σx·Σy) / (n·Σ(x²) − (Σx)²)
        slope_pct_per_hour = (slope_per_minute / mean_price) × 100 × 60

        Args:
            closes: Full close-price list.
            window: Number of trailing closes to regress over.

        Returns:
            Slope in % per hour.
        """
        segment = closes[-window:]
        n = len(segment)
        if n < 2:
            return 0.0

        sum_x = 0.0
        sum_y = 0.0
        sum_xy = 0.0
        sum_x2 = 0.0
        for i in range(n):
            x = float(i)
            y = segment[i]
            sum_x += x
            sum_y += y
            sum_xy += x * y
            sum_x2 += x * x

        denom = n * sum_x2 - sum_x * sum_x
        if denom == 0.0:
            return 0.0

        slope_per_minute = (n * sum_xy - sum_x * sum_y) / denom
        mean_price = sum_y / n
        if mean_price == 0.0:
            return 0.0

        return (slope_per_minute / mean_price) * 100.0 * 60.0

    @staticmethod
    def _compute_band(highs: List[float], lows: List[float]) -> float:
        """24-hour band: (max_high − min_low) / min_low.

        Args:
            highs: High prices.
            lows: Low prices.

        Returns:
            Band as a fraction (e.g. 0.05 = 5%).
        """
        if not highs or not lows:
            return 0.0
        max_high = max(highs)
        min_low = min(lows)
        if min_low == 0.0:
            return 0.0
        return (max_high - min_low) / min_low

    def _determine_regime(self, atr: float, slope: float, band: float) -> str:
        """Applies the classification rules to indicator values.

        Args:
            atr: ATR-14 value.
            slope: Slope in % per hour.
            band: 24h band fraction.

        Returns:
            One of ``TRENDING_HARD``, ``SLOW_BULL``, ``SLOW_BEAR``, ``RANGING``.
        """
        if (
            atr > self._atr_trending
            or abs(slope) > self._slope_hard
            or band > 0.08
        ):
            return "TRENDING_HARD"
        elif slope > self._slope_bull and atr < self._atr_trending:
            return "SLOW_BULL"
        elif slope < self._slope_bear and atr < self._atr_trending:
            return "SLOW_BEAR"
        else:
            return "RANGING"
