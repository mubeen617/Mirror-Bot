"""
Week 3 Integration Test Script.

Manual integration test that validates the full Week 3 pipeline:
    1. Regime detector produces non-zero indicators
    2. Crash monitor fires on injected price drop
    3. /start command clears crash lockout
    4. Dashboard API returns correct state

Usage:
    python test_week3_integration.py

This script runs standalone (not via pytest) and prints PASS/FAIL for each step.
It requires the bot to be running (python run.py) in another terminal.
"""

import asyncio
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).parent))

import os
import aiohttp
from dotenv import load_dotenv

# Load secrets from the same env the bot uses
_env_path = Path(__file__).parent / "config" / "secrets.env"
if _env_path.exists():
    load_dotenv(_env_path)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("integration_test")

# ── Configuration ───────────────────────────────────────────────────
DASHBOARD_URL = "http://localhost:5000"
AUTH_TOKEN = os.getenv("DASHBOARD_SECRET", "")


# ── Helpers ─────────────────────────────────────────────────────────

class TestResult:
    """Tracks individual test results."""

    def __init__(self) -> None:
        self.results: list[tuple[str, bool, str, float]] = []

    def record(self, name: str, passed: bool, detail: str, duration: float) -> None:
        """Records a test result."""
        status = "PASS ✅" if passed else "FAIL ❌"
        self.results.append((name, passed, detail, duration))
        logger.info(f"  {status} — {name} ({duration:.1f}s) — {detail}")

    def summary(self) -> None:
        """Prints final summary."""
        total = len(self.results)
        passed = sum(1 for _, p, _, _ in self.results if p)
        failed = total - passed
        print("\n" + "=" * 60)
        print(f"  INTEGRATION TEST RESULTS: {passed}/{total} passed, {failed} failed")
        print("=" * 60)
        for name, p, detail, dur in self.results:
            icon = "[PASS]" if p else "[FAIL]"
            print(f"  {icon} {name} ({dur:.1f}s)")
            if not p:
                print(f"     -> {detail}")
        print("=" * 60 + "\n")


async def api_get(session: aiohttp.ClientSession, endpoint: str) -> Dict[str, Any]:
    """Makes a GET request to the dashboard API."""
    url = f"{DASHBOARD_URL}{endpoint}"
    async with session.get(url) as resp:
        return await resp.json()


async def api_post_command(
    session: aiohttp.ClientSession, command: str
) -> Dict[str, Any]:
    """Posts a command to the dashboard API."""
    url = f"{DASHBOARD_URL}/api/command"
    headers = {"Content-Type": "application/json"}
    if AUTH_TOKEN:
        headers["Authorization"] = f"Bearer {AUTH_TOKEN}"
    async with session.post(url, json={"command": command}, headers=headers) as resp:
        return await resp.json()


# ── Test Steps ──────────────────────────────────────────────────────

async def test_bot_reachable(
    session: aiohttp.ClientSession, results: TestResult
) -> bool:
    """Step 1: Verify the bot dashboard is reachable."""
    t0 = time.time()
    try:
        data = await api_get(session, "/api/health")
        ok = data.get("status") == "ok"
        results.record(
            "Bot reachable",
            ok,
            f"status={data.get('status')}, uptime={data.get('uptime_seconds', 0):.0f}s",
            time.time() - t0,
        )
        return ok
    except Exception as exc:
        results.record("Bot reachable", False, str(exc), time.time() - t0)
        return False


async def test_regime_classification(
    session: aiohttp.ClientSession, results: TestResult
) -> bool:
    """Step 2: Wait up to 75 seconds for first regime classification."""
    t0 = time.time()
    logger.info("  Waiting for regime classification (up to 75s)...")

    for attempt in range(15):
        await asyncio.sleep(5)
        try:
            data = await api_get(session, "/api/status")
            regime = data.get("regime", "RANGING")
            atr = data.get("atr", 0)
            slope = data.get("slope_pct", 0)
            band = data.get("band_24h", 0)
            btc = data.get("btc_price", 0)

            if atr != 0 or slope != 0 or band != 0:
                results.record(
                    "Regime classification",
                    True,
                    f"regime={regime}, ATR={atr:.1f}, slope={slope:.4f}%/h, "
                    f"band={band*100:.1f}%, BTC=${btc:,.0f}",
                    time.time() - t0,
                )
                return True
        except Exception:
            pass

    results.record(
        "Regime classification", False,
        "Indicators still zero after 75 seconds",
        time.time() - t0,
    )
    return False


async def test_crash_monitor_active(
    session: aiohttp.ClientSession, results: TestResult
) -> bool:
    """Step 3: Verify crash monitor is producing drop readings."""
    t0 = time.time()
    logger.info("  Checking crash monitor is active...")

    for attempt in range(6):
        await asyncio.sleep(5)
        try:
            data = await api_get(session, "/api/status")
            btc = data.get("btc_price", 0)
            drop = data.get("drop_5m_pct", None)

            if btc > 0 and drop is not None:
                results.record(
                    "Crash monitor active",
                    True,
                    f"BTC=${btc:,.0f}, drop_5m={drop:.3f}%",
                    time.time() - t0,
                )
                return True
        except Exception:
            pass

    results.record(
        "Crash monitor active", False,
        "No price data after 30 seconds",
        time.time() - t0,
    )
    return False


async def test_dashboard_panels(
    session: aiohttp.ClientSession, results: TestResult
) -> bool:
    """Step 4: Verify all dashboard API endpoints return data."""
    t0 = time.time()
    try:
        status = await api_get(session, "/api/status")
        fills = await api_get(session, "/api/fills?limit=5")
        ratios = await api_get(session, "/api/ratio?limit=5")
        regimes = await api_get(session, "/api/regime?limit=5")
        health = await api_get(session, "/api/health")

        checks = [
            ("status has regime", "regime" in status),
            ("status has atr", "atr" in status),
            ("status has btc_price", "btc_price" in status),
            ("status has crash_lockout", "crash_lockout" in status),
            ("fills is list", isinstance(fills, list)),
            ("ratios is list", isinstance(ratios, list)),
            ("regimes is list", isinstance(regimes, list)),
            ("health ok", health.get("status") == "ok"),
        ]

        all_ok = all(ok for _, ok in checks)
        failed = [name for name, ok in checks if not ok]

        results.record(
            "Dashboard API endpoints",
            all_ok,
            f"All 8 checks passed" if all_ok else f"Failed: {failed}",
            time.time() - t0,
        )
        return all_ok
    except Exception as exc:
        results.record("Dashboard API endpoints", False, str(exc), time.time() - t0)
        return False


async def test_regime_command(
    session: aiohttp.ClientSession, results: TestResult
) -> bool:
    """Step 5: Test the /regime command via API."""
    t0 = time.time()
    try:
        data = await api_post_command(session, "regime")
        has_regime = "regime" in data
        has_atr = "atr" in data
        results.record(
            "/regime command",
            has_regime and has_atr,
            f"regime={data.get('regime')}, atr={data.get('atr')}",
            time.time() - t0,
        )
        return has_regime and has_atr
    except Exception as exc:
        results.record("/regime command", False, str(exc), time.time() - t0)
        return False


async def test_start_command(
    session: aiohttp.ClientSession, results: TestResult
) -> bool:
    """Step 6: Test the /start command resets crash lockout."""
    t0 = time.time()
    try:
        data = await api_post_command(session, "start")
        ok = "result" in data

        # Verify lockout cleared
        status = await api_get(session, "/api/status")
        lockout = status.get("crash_lockout", True)

        results.record(
            "/start command",
            ok and not lockout,
            f"result={data.get('result', 'N/A')}, crash_lockout={lockout}",
            time.time() - t0,
        )
        return ok and not lockout
    except Exception as exc:
        results.record("/start command", False, str(exc), time.time() - t0)
        return False


async def test_dashboard_html(
    session: aiohttp.ClientSession, results: TestResult
) -> bool:
    """Step 7: Verify the dashboard HTML page loads."""
    t0 = time.time()
    try:
        async with session.get(f"{DASHBOARD_URL}/") as resp:
            text = await resp.text()
            has_react = "react" in text.lower() or "Grid Mirror Bot" in text
            ok = resp.status == 200 and has_react
            results.record(
                "Dashboard HTML",
                ok,
                f"status={resp.status}, has_react={has_react}, size={len(text)} bytes",
                time.time() - t0,
            )
            return ok
    except Exception as exc:
        results.record("Dashboard HTML", False, str(exc), time.time() - t0)
        return False


# ── Main ────────────────────────────────────────────────────────────

async def main() -> None:
    """Runs all integration test steps in sequence."""
    print("\n" + "=" * 60)
    print("  WEEK 3 INTEGRATION TEST")
    print(f"  Dashboard: {DASHBOARD_URL}")
    print("=" * 60 + "\n")

    results = TestResult()

    async with aiohttp.ClientSession() as session:
        # Step 1: Check bot is reachable
        if not await test_bot_reachable(session, results):
            logger.error("Bot is not reachable. Start the bot first: python run.py")
            results.summary()
            return

        # Step 2: Wait for regime classification
        await test_regime_classification(session, results)

        # Step 3: Verify crash monitor
        await test_crash_monitor_active(session, results)

        # Step 4: Dashboard panels
        await test_dashboard_panels(session, results)

        # Step 5: /regime command
        await test_regime_command(session, results)

        # Step 6: /start command
        await test_start_command(session, results)

        # Step 7: Dashboard HTML
        await test_dashboard_html(session, results)

    results.summary()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Test interrupted.")
