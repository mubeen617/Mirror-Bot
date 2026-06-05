"""
Mock BUY Trade Simulation Script.
Allows testing the BUY order flow through the entire grid bot pipeline
(Compounding, SQLite database, Telegram/WhatsApp Alerters, and MT5 Executor)
"""

import asyncio
import sys
import time
from pathlib import Path

# Add parent directory to path so we can import modules
sys.path.append(str(Path(__file__).parent))

from core.db import DatabaseManager
from core.compounding import CompoundingEngine
from alerts.telegram import TelegramAlerter
from alerts.whatsapp import WhatsAppAlerter
from run import load_config


def fetch_live_price(symbol: str = "BTCUSDT") -> float:
    """Attempts to fetch the live price from the running bot's status API or public Binance API.
    
    Falls back to a default value if both fail.
    """
    import urllib.request
    import json
    # 1. Try bot status API
    try:
        url = "http://127.0.0.1:5000/api/status"
        with urllib.request.urlopen(url, timeout=3.0) as response:
            data = json.loads(response.read().decode("utf-8"))
            price = float(data.get("btc_price", 0.0))
            if price > 0.0:
                print(f"Fetched live price from running bot status API: ${price:,.2f}")
                return price
    except Exception:
        pass

    # 2. Try public Binance API
    try:
        url = f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}"
        with urllib.request.urlopen(url, timeout=3.0) as response:
            data = json.loads(response.read().decode("utf-8"))
            price = float(data.get("price", 0.0))
            if price > 0.0:
                print(f"Fetched live price from public Binance API: ${price:,.2f}")
                return price
    except Exception:
        pass

    # 3. Fallback price
    fallback = 95420.0
    print(f"Could not fetch live price. Using fallback: ${fallback:,.2f}")
    return fallback


async def run_simulation() -> None:
    """Executes a simulated BUY fill through the live bot API or local offline pipeline."""
    print("=======================================")
    print("  GRID MIRROR BOT - Mock BUY Simulation")
    print("=======================================")

    # 1. Load configurations
    config = load_config()
    symbol = config.get("binance", {}).get("symbol", "BTCUSDT")

    # Try connecting to the running bot process first via HTTP API
    import json
    import urllib.request
    from urllib.error import URLError

    dashboard_cfg = config.get("dashboard", {})
    host = dashboard_cfg.get("host", "127.0.0.1")
    if host == "0.0.0.0":
        host = "127.0.0.1"
    port = dashboard_cfg.get("port", 5000)
    secret = config.get("DASHBOARD_SECRET", "")

    url = f"http://{host}:{port}/api/simulate"
    headers = {
        "Authorization": f"Bearer {secret}",
        "Content-Type": "application/json",
    }
    
    # Fetch live price dynamically to avoid artificial drawdown
    mock_price = fetch_live_price(symbol)
    
    payload = {
        "price": mock_price,
        "qty": 0.0001,
        "side": "BUY",
        "symbol": symbol,
    }

    print(f"Checking for live bot running on {host}:{port}...")
    try:
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST"
        )
        # Timeout of 90 seconds to allow full MT5 deal execution
        with urllib.request.urlopen(req, timeout=90.0) as response:
            res_data = json.loads(response.read().decode("utf-8"))
            print("\n[SUCCESS] Live bot detected! Successfully sent simulated BUY fill to the running bot.")
            print(f"Response: {res_data.get('result')}")
            print("\n=======================================")
            print("  Simulation completed successfully!  ")
            print("=======================================")
            return
    except urllib.error.HTTPError as e:
        print(f"\n[ERROR] Live bot returned error code {e.code}: {e.reason}")
        try:
            error_body = e.read().decode("utf-8")
            print(f"Response details: {error_body}")
        except Exception:
            pass
        print("Running offline simulation mode...")
    except URLError as e:
        print(f"\n[ERROR] Live bot not detected or not responding on {host}:{port} ({e}).")
        print("Running offline simulation mode...")

    # 2. Initialize database
    root_dir = Path(__file__).parent
    db_manager = DatabaseManager(root_dir / "db" / "trades.db")
    await db_manager.initialize()

    # 3. Initialize compounding and alerter services
    compounding_engine = CompoundingEngine(config, db_manager)
    
    # We set a mock balance and smoothed ratio for the simulation
    compounding_engine.binance_balance = 100.0
    compounding_engine.smoothed_ratio = 100.0  # 100x scale factor
    compounding_engine.baseline_balance = 100.0

    telegram_alerter = TelegramAlerter(config)
    await telegram_alerter.start()

    whatsapp_alerter = WhatsAppAlerter(config)
    await whatsapp_alerter.start()

    # 4. Generate mock fill dictionary using the dynamically fetched price
    mock_qty = 0.0001
    now_ms = int(time.time() * 1000)

    print(f"\n1. Generating mock BUY trade: BUY {mock_qty} {symbol} @ ${mock_price:,.2f}")
    mock_fill = {
        "symbol": symbol,
        "side": "BUY",
        "qty": mock_qty,
        "price": mock_price,
        "order_id": f"MOCK_BUY_FILL_{int(time.time())}",
        "timestamp": now_ms,
    }

    # 5. Process fill via Compounding Engine
    print("2. Calculating scaled order sizing via CompoundingEngine...")
    scaled = compounding_engine.process_fill(mock_fill)
    print(f"   -> Smoothed Ratio: {scaled['scale_ratio']:.1f}x")
    print(f"   -> Scaled Quantity: {scaled['scaled_qty']:.4f} {scaled['symbol']}")
    print(f"   -> Kill Switch: {'ON' if scaled['kill_switch'] else 'OFF'} ({scaled['reason']})")

    # 6. Database log write
    print("3. Logging execution fill and calculations asynchronously to SQLite...")
    await db_manager.log_fill(
        timestamp=mock_fill["timestamp"],
        symbol=mock_fill["symbol"],
        side=mock_fill["side"],
        binance_qty=mock_fill["qty"],
        binance_price=mock_fill["price"],
        order_id=mock_fill["order_id"],
        scale_ratio=scaled["scale_ratio"],
        scaled_qty=scaled["scaled_qty"],
        kill_switch=scaled["kill_switch"],
        kill_reason=scaled["reason"] if scaled["kill_switch"] else None,
    )
    print("   -> SQLite write completed successfully.")

    # 7. Dispatch Alerts
    alerts_waiting = False

    if telegram_alerter.enabled:
        print("4. Dispatching scaled fill alert to Telegram Queue...")
        await telegram_alerter.send_fill(mock_fill, scaled)
        alerts_waiting = True
    else:
        print("4. Telegram is disabled or credentials missing. Skipping Telegram notification.")

    if whatsapp_alerter.enabled:
        print("5. Dispatching scaled fill alert to WhatsApp Queue...")
        await whatsapp_alerter.send_fill(mock_fill, scaled)
        alerts_waiting = True
    else:
        print("5. WhatsApp is disabled or credentials missing. Skipping WhatsApp notification.")

    # Wait a moment for queue workers to dispatch
    if alerts_waiting:
        await asyncio.sleep(2.0)

    # 8. Cleanup
    await telegram_alerter.stop()
    await whatsapp_alerter.stop()
    print("\n=======================================")
    print("  Simulation completed successfully!  ")
    print("=======================================")


if __name__ == "__main__":
    try:
        asyncio.run(run_simulation())
    except KeyboardInterrupt:
        print("\nSimulation aborted.")
