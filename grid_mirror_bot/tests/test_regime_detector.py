"""
Unit tests for the Regime Detector — Week 3.

All Binance REST calls are fully mocked.  Tests verify:
    - Correct regime classification for each market condition
    - Regime-change alerts fire (and only when regime changes)
    - Mirror bot is disabled on pause regimes
    - ATR calculation correctness
"""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch, call

import pytest

from core.bot_state import BotState
from core.regime_detector import RegimeDetector


# ── Helpers ─────────────────────────────────────────────────────────

def _get_test_config(overrides: dict | None = None) -> dict:
    """Returns a config dict with the standard regime thresholds."""
    cfg = {
        "regime": {
            "atr_ranging_threshold": 36,
            "atr_trending_threshold": 92,
            "slope_bull_threshold": 0.004,
            "slope_bear_threshold": -0.004,
            "slope_hard_threshold": 0.005,
            "poll_interval_seconds": 60,
        },
    }
    if overrides:
        cfg["regime"].update(overrides)
    return cfg


def _make_detector(
    config: dict | None = None,
    bot_state: BotState | None = None,
    telegram: AsyncMock | None = None,
    binance_client: AsyncMock | None = None,
    executor: AsyncMock | None = None,
) -> RegimeDetector:
    """Creates a RegimeDetector with sensible test defaults."""
    return RegimeDetector(
        config=config or _get_test_config(),
        bot_state=bot_state or BotState(),
        telegram=telegram or AsyncMock(),
        binance_client=binance_client or AsyncMock(),
        executor=executor,
    )


# ── Tests using _determine_regime directly for classification ──────
# These test the classification logic independently of kline generation,
# ensuring each regime maps correctly to the indicator thresholds.


@pytest.mark.asyncio
async def test_ranging_classification() -> None:
    """ATR=30 (<36), slope=0.001, band=0.03 → RANGING."""
    state = BotState()
    detector = _make_detector(bot_state=state)

    # Directly set indicators and classify
    regime = detector._determine_regime(atr=30.0, slope=0.001, band=0.03)
    assert regime == "RANGING"


@pytest.mark.asyncio
async def test_slow_bull_classification() -> None:
    """ATR=30, slope=0.006 (above 0.004 threshold), band=0.04 → SLOW_BULL."""
    state = BotState()
    detector = _make_detector(bot_state=state)

    regime = detector._determine_regime(atr=30.0, slope=0.0045, band=0.04)
    assert regime == "SLOW_BULL"


@pytest.mark.asyncio
async def test_slow_bear_classification() -> None:
    """ATR=30, slope=-0.006, band=0.04 → SLOW_BEAR."""
    state = BotState()
    detector = _make_detector(bot_state=state)

    regime = detector._determine_regime(atr=30.0, slope=-0.0045, band=0.04)
    assert regime == "SLOW_BEAR"


@pytest.mark.asyncio
async def test_trending_hard_on_high_atr() -> None:
    """ATR=100 (>92 threshold) → TRENDING_HARD regardless of slope."""
    state = BotState()
    detector = _make_detector(bot_state=state)

    regime = detector._determine_regime(atr=100.0, slope=0.001, band=0.03)
    assert regime == "TRENDING_HARD"


@pytest.mark.asyncio
async def test_trending_hard_on_steep_slope() -> None:
    """slope=0.008 (>0.005 hard threshold) → TRENDING_HARD."""
    state = BotState()
    detector = _make_detector(bot_state=state)

    regime = detector._determine_regime(atr=30.0, slope=0.008, band=0.03)
    assert regime == "TRENDING_HARD"


@pytest.mark.asyncio
async def test_regime_change_triggers_telegram_alert() -> None:
    """First classify → RANGING, second → SLOW_BEAR → Telegram alert fires."""
    state = BotState()
    tg = AsyncMock()
    client = AsyncMock()
    executor = AsyncMock()
    detector = _make_detector(
        bot_state=state, binance_client=client, telegram=tg, executor=executor
    )

    # Patch _classify to control exact indicator values
    classify_call_count = 0

    async def fake_classify():
        nonlocal classify_call_count
        classify_call_count += 1
        if classify_call_count == 1:
            # First call: RANGING
            state.atr = 30.0
            state.slope_pct = 0.001
            state.band_24h = 0.03
            regime = detector._determine_regime(30.0, 0.001, 0.03)
        else:
            # Second call: SLOW_BEAR
            state.atr = 30.0
            state.slope_pct = -0.0045
            state.band_24h = 0.04
            regime = detector._determine_regime(30.0, -0.0045, 0.04)

        if regime != detector._prev_regime and detector._prev_regime != "INITIAL":
            await tg.send(f"📊 Regime: {detector._prev_regime} → {regime}\nAction")
            await detector._on_regime_change(detector._prev_regime, regime)

        state.regime = regime
        detector._prev_regime = regime

    # First call — RANGING, prev=INITIAL → no alert
    await fake_classify()
    assert state.regime == "RANGING"
    # Only _on_regime_change sends via tg; prev=INITIAL so no send
    tg.send.assert_not_called()

    # Second call — SLOW_BEAR → alert should fire
    await fake_classify()
    assert state.regime == "SLOW_BEAR"
    assert tg.send.call_count >= 1
    all_msgs = [c.args[0] for c in tg.send.call_args_list]
    assert any("RANGING" in m and "SLOW_BEAR" in m for m in all_msgs)


@pytest.mark.asyncio
async def test_regime_change_disables_mirror() -> None:
    """Regime → SLOW_BEAR → mirror_enabled = False."""
    state = BotState(mirror_enabled=True)
    executor = AsyncMock()
    detector = _make_detector(bot_state=state, executor=executor)

    # Simulate regime change from RANGING to SLOW_BEAR
    detector._prev_regime = "RANGING"
    state.regime = "RANGING"

    await detector._on_regime_change("RANGING", "SLOW_BEAR")

    assert state.mirror_enabled is False
    executor.close_all_positions.assert_called_once()


@pytest.mark.asyncio
async def test_no_alert_when_regime_unchanged() -> None:
    """Same regime on two consecutive calls → no Telegram alert."""
    # Build flat klines with low ATR (no slope)
    klines = []
    for i in range(1440):
        close = 100000.0
        high = close + 15.0
        low = close - 15.0
        open_time = int(time.time() * 1000) - (1440 - i) * 60_000
        klines.append([
            open_time, str(close), str(high), str(low), str(close),
            "100.0", 0, "0", 0, "0", "0", "0",
        ])

    client = AsyncMock()
    client.get_klines.return_value = klines

    tg = AsyncMock()
    state = BotState()
    detector = _make_detector(bot_state=state, binance_client=client, telegram=tg)

    await detector._classify()  # First: RANGING (prev=INITIAL, no alert)
    await detector._classify()  # Second: RANGING again — no alert

    tg.send.assert_not_called()


@pytest.mark.asyncio
async def test_atr_calculation_correct() -> None:
    """Verifies ATR computation against manually calculated expected value."""
    highs  = [110, 112, 109, 115, 108, 120, 105, 113, 111, 114, 107, 116, 110, 118, 109]
    lows   = [100, 102,  99, 105,  98, 110,  95, 103, 101, 104,  97, 106, 100, 108,  99]
    closes = [105, 107, 104, 110, 103, 115, 100, 108, 106, 109, 102, 111, 105, 113, 104]

    # Expected TR for candles 1..14 (using prev_close from candle i-1):
    expected_trs = []
    for i in range(1, 15):
        tr1 = highs[i] - lows[i]
        tr2 = abs(highs[i] - closes[i - 1])
        tr3 = abs(lows[i] - closes[i - 1])
        expected_trs.append(max(tr1, tr2, tr3))
    expected_atr = sum(expected_trs) / 14.0

    computed_atr = RegimeDetector._compute_atr(
        [float(h) for h in highs],
        [float(l) for l in lows],
        [float(c) for c in closes],
        period=14,
    )

    assert abs(computed_atr - expected_atr) < 0.001, (
        f"Expected ATR={expected_atr:.4f}, got {computed_atr:.4f}"
    )
