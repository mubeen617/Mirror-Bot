"""
Main entry point for the Grid Mirror Bot.
Performs package auto-installation, loads configurations, handles signal termination,
and coordinates all async engine tasks.
"""

import sys
import subprocess
from pathlib import Path

# 1. Dependency Auto-Installation
REQUIRED_PACKAGES = [
    ("binance", "python-binance"),
    ("yaml", "pyyaml"),
    ("dotenv", "python-dotenv"),
    ("aiofiles", "aiofiles"),
    ("telegram", "python-telegram-bot"),
    ("aiosqlite", "aiosqlite"),
    ("MetaTrader5", "MetaTrader5"),
    ("flask", "flask"),
    ("flask_cors", "flask-cors"),
]

missing = []
for module_name, package_name in REQUIRED_PACKAGES:
    try:
        __import__(module_name)
    except ImportError:
        missing.append(package_name)

if missing:
    print(f"Missing required packages: {missing}. Auto-installing...")
    req_file = Path(__file__).parent / "requirements.txt"
    try:
        if req_file.exists():
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(req_file)])
        else:
            subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing)
        print("Auto-installation completed successfully. Launching bot...")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred during auto-installation: {e}")
        sys.exit(1)

# Now standard imports
import argparse
import asyncio
import logging
import signal
import time
from typing import Any, Dict, List
import yaml
from dotenv import load_dotenv

# App imports
from core.db import DatabaseManager
from core.compounding import CompoundingEngine
from core.grid_simulator import GridSimulator
from core.watcher import BinanceWatcher
from core.bot_state import BotState
from core.executor import MT5Executor
from core.reconciler import Reconciler
from core.regime_detector import RegimeDetector
from core.crash_monitor import CrashMonitor
from core.pair_state import PairState
from core.pair_worker import PairWorker
from core.pair_watcher import PairWatcher
from core.symbol_validator import SymbolValidator
from core.health_monitor import HealthMonitor
from alerts.telegram import TelegramAlerter
from alerts.whatsapp import WhatsAppAlerter
from dashboard.server import start_dashboard_thread
from binance import AsyncClient

# Root logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("run")


def load_config() -> Dict[str, Any]:
    """Loads configuration yaml and secrets from env."""
    root_path = Path(__file__).parent
    config_file = root_path / "config" / "config.yaml"
    
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    # Load secrets
    env_file = root_path / "config" / "secrets.env"
    if not env_file.exists():
        # Fallback to general .env
        env_file = root_path / ".env"
    
    load_dotenv(dotenv_path=env_file)
    import os
    config["BINANCE_API_KEY"] = os.getenv("BINANCE_API_KEY")
    config["BINANCE_API_SECRET"] = os.getenv("BINANCE_API_SECRET")
    config["TELEGRAM_BOT_TOKEN"] = os.getenv("TELEGRAM_BOT_TOKEN")
    config["TELEGRAM_CHAT_ID"] = os.getenv("TELEGRAM_CHAT_ID")
    config["DASHBOARD_SECRET"] = os.getenv("DASHBOARD_SECRET")
    config["WHATSAPP_PHONE"] = os.getenv("WHATSAPP_PHONE")
    config["WHATSAPP_API_KEY"] = os.getenv("WHATSAPP_API_KEY")
    config["WHATSAPP_ACCOUNT_SID"] = os.getenv("WHATSAPP_ACCOUNT_SID")
    config["WHATSAPP_AUTH_TOKEN"] = os.getenv("WHATSAPP_AUTH_TOKEN")

    # MT5 secrets
    mt5_cfg = config.get("mt5", {})
    mt5_login = os.getenv("MT5_LOGIN", mt5_cfg.get("login", ""))
    mt5_cfg["login"] = int(mt5_login) if str(mt5_login).isdigit() else 0
    mt5_cfg["password"] = os.getenv("MT5_PASSWORD", mt5_cfg.get("password", ""))
    mt5_cfg["server"] = os.getenv("MT5_SERVER", mt5_cfg.get("server", ""))
    config["mt5"] = mt5_cfg

    return config


def validate_secrets(config: Dict[str, Any]) -> None:
    """Ensures necessary Binance and Dashboard keys are loaded."""
    missing_keys = []
    if not config.get("BINANCE_API_KEY"):
        missing_keys.append("BINANCE_API_KEY")
    if not config.get("BINANCE_API_SECRET"):
        missing_keys.append("BINANCE_API_SECRET")
    
    if missing_keys:
        logger.error(f"CRITICAL: Missing environment secrets in secrets.env: {missing_keys}")
        sys.exit(1)


# NOTE: The inline run_regime_detector() and run_crash_monitor() functions
# from Week 2 have been replaced by the class-based RegimeDetector and
# CrashMonitor modules in core/regime_detector.py and core/crash_monitor.py.


async def run_compounding_poller(compounding_engine: CompoundingEngine, client: AsyncClient, interval: int) -> None:
    """Background task updating risk metrics and smoothed scaling ratios."""
    logger.info("Compounding engine poller started.")
    while True:
        await compounding_engine.poll(client)
        await asyncio.sleep(interval)


class BotRunner:
    """Manages the full lifecycle of the bot components."""

    def __init__(self, config: Dict[str, Any], args: argparse.Namespace) -> None:
        self.config = config
        self.args = args
        self.client: AsyncClient | None = None
        self.db_manager: DatabaseManager | None = None
        self.compounding_engine: CompoundingEngine | None = None
        self.grid_simulator: GridSimulator | None = None
        self.watcher: BinanceWatcher | None = None
        self.telegram_alerter: TelegramAlerter | None = None
        self.whatsapp_alerter: WhatsAppAlerter | None = None
        self.bot_state: BotState = BotState(session_start=time.time())
        self.executor: MT5Executor | None = None
        self.reconciler: Reconciler | None = None
        self.regime_detector: RegimeDetector | None = None
        self.crash_monitor: CrashMonitor | None = None
        self.tasks: List[asyncio.Task] = []
        self.shutdown_event = asyncio.Event()

        # Week 4: Multi-pair components
        self.pair_states: Dict[str, PairState] = {}
        self.pair_workers: Dict[str, PairWorker] = {}
        self.pair_watcher: PairWatcher | None = None
        self.health_monitor: HealthMonitor | None = None
        self.validation_results: Dict[str, Any] = {}
        self._multi_pair_enabled = config.get("multi_pair", {}).get("enabled", False)

    def print_banner(self) -> None:
        """Prints high-quality startup console header."""
        binance_env = "TESTNET" if self.config.get("binance", {}).get("testnet", True) else "MAINNET"
        mt5_mode = "DEMO" if self.config.get("mt5", {}).get("demo_mode", True) else "LIVE"

        mt5_server = self.config.get("mt5", {}).get("server", "")
        mt5_broker = "MT5"
        if "icmarkets" in mt5_server.lower():
            mt5_broker = "ICMarkets"
        elif "fundednext" in mt5_server.lower():
            mt5_broker = "FundedNext"
        elif mt5_server:
            mt5_broker = mt5_server

        if self._multi_pair_enabled and self.pair_states:
            pairs_str = " | ".join(
                f"{ps.pair} -> {ps.mt5_symbol}"
                for ps in self.pair_states.values()
            )
            banner = f"""
═══════════════════════════════════════════════════════
  GRID MIRROR BOT — Week 4 Multi-Pair
  Binance:  {binance_env}
  MT5:      {mt5_broker} {mt5_mode} (swap to FundedNext when ready)
  Pairs:    {pairs_str}
  Alerts:   WhatsApp
═══════════════════════════════════════════════════════"""
            print(banner)

            print(f"  {'Pair':<12} {'MT5 Symbol':<12} {'Capital':<10} {'Mirror':<9} {'Regime'}")
            print(f"  {'─' * 55}")
            for ps in self.pair_states.values():
                mirror_str = "ON" if ps.mirror_enabled else "OFF"
                print(
                    f"  {ps.pair:<12} {ps.mt5_symbol:<12} "
                    f"${ps.grid_capital:<9.2f} {mirror_str:<9} {ps.regime}"
                )
            print("═" * 55)
        else:
            banner = f"""
═══════════════════════════════════════════════
  GRID MIRROR BOT — Week 3 Full System
  Binance:  {binance_env}
  MT5:      {mt5_mode} — {mt5_broker}
  Regime:   Starting up...
  Crash:    Monitoring every 5 seconds
  Dashboard: http://localhost:5000
  Alerts:   WhatsApp + Telegram
═══════════════════════════════════════════════
"""
            print(banner)

    async def shutdown(self, sig: signal.Signals | None = None) -> None:
        """Gracefully tears down tasks, cancels grid orders, closes MT5, and alerts Telegram."""
        if sig:
            logger.info(f"Received exit signal {sig.name}...")
        else:
            logger.info("Initiating graceful shutdown...")

        # Stop multi-pair components
        if self.pair_watcher:
            await self.pair_watcher.stop()
        for worker in self.pair_workers.values():
            await worker.stop()

        # Stop watchers and alert system
        if self.watcher:
            await self.watcher.stop()

        # Close all MT5 positions before shutting down
        if self.executor:
            logger.info("Closing all MT5 positions on shutdown...")
            try:
                await self.executor.close_all_positions(reason="shutdown")
            except Exception as e:
                logger.error(f"Error closing MT5 positions on shutdown: {e}")
            await self.executor.stop()

        # Stop reconciler
        if self.reconciler:
            await self.reconciler.stop()

        # Cancel all grid orders
        if self.client and self.grid_simulator:
            logger.info("Cancelling all open grid orders on Testnet...")
            await self.grid_simulator.cancel_all(self.client, self.config.get("binance", {}).get("symbol", "BTCUSDT"))

        # Send Telegram shutdown alert
        if self.telegram_alerter:
            await self.telegram_alerter.send("🤖 Bot shutting down gracefully.")
            await self.telegram_alerter.stop()

        # Send WhatsApp shutdown alert
        if self.whatsapp_alerter:
            await self.whatsapp_alerter.send("🤖 Bot shutting down gracefully.")
            await self.whatsapp_alerter.stop()

        # Cancel polling/monitoring tasks
        for task in self.tasks:
            task.cancel()
        
        if self.tasks:
            await asyncio.gather(*self.tasks, return_exceptions=True)

        if self.db_manager:
            await self.db_manager.close()

        if self.client:
            await self.client.close_connection()

        self.shutdown_event.set()
        logger.info("Graceful shutdown completed successfully.")

    async def execute(self) -> None:
        """Instantiates services, schedules asyncio loops, and binds signals."""
        try:
            # Print header
            self.print_banner()

            # Initialize clients
            self.client = await AsyncClient.create(
                api_key=self.config["BINANCE_API_KEY"],
                api_secret=self.config["BINANCE_API_SECRET"],
                testnet=self.config.get("binance", {}).get("testnet", True),
            )

            # Initialize DB
            root_dir = Path(__file__).parent
            self.db_manager = DatabaseManager(root_dir / "db" / "trades.db")
            await self.db_manager.initialize()

            # Disable Telegram if flag passed
            if self.args.no_telegram:
                self.config["telegram"]["enabled"] = False

            # Initialize Services
            self.telegram_alerter = TelegramAlerter(self.config)
            await self.telegram_alerter.start()

            self.whatsapp_alerter = WhatsAppAlerter(self.config)
            await self.whatsapp_alerter.start()
            
            self.compounding_engine = CompoundingEngine(self.config, self.db_manager)
            self.bot_state.smoothed_ratio = self.compounding_engine.smoothed_ratio
            self.bot_state.scale_ratio = self.compounding_engine.smoothed_ratio
            self.bot_state.binance_balance = self.compounding_engine.binance_balance
            self.grid_simulator = GridSimulator()

            # ── Week 2: MT5 Executor ─────────────────────────────
            self.executor = MT5Executor(
                self.config, self.bot_state, self.telegram_alerter
            )
            try:
                await self.executor.start()
                logger.info("MT5 executor started successfully.")
            except RuntimeError as e:
                logger.error(f"MT5 executor failed to start: {e}")
                await self.telegram_alerter.send(f"❌ MT5 executor failed to start: {e}")
                # Continue without MT5 — executor will remain disconnected

            # ── Week 4: Multi-Pair Setup ─────────────────────────
            if self._multi_pair_enabled:
                await self._setup_multi_pair()

            # ── Week 2: Wire executor into watcher fill callback ─
            executor_ref = self.executor
            bot_state_ref = self.bot_state

            async def _fill_callback_with_mt5(fill: Dict[str, Any]) -> None:
                """Extended fill callback — scales via compounding, then mirrors to MT5."""
                try:
                    scaled = self.compounding_engine.process_fill(fill)
                    from core.watcher import watcher_logger
                    watcher_logger.info(
                        f"SCALED ORDER | Ratio: {scaled['scale_ratio']:.2f}x | "
                        f"Qty: {scaled['scaled_qty']:.4f} | Kill: {scaled['kill_switch']} | "
                        f"Reason: {scaled['reason']}"
                    )

                    # Database write
                    if self.db_manager:
                        await self.db_manager.log_fill(
                            timestamp=fill["timestamp"],
                            symbol=fill["symbol"],
                            side=fill["side"],
                            binance_qty=fill["qty"],
                            binance_price=fill["price"],
                            order_id=fill["order_id"],
                            scale_ratio=scaled["scale_ratio"],
                            scaled_qty=scaled["scaled_qty"],
                            kill_switch=scaled["kill_switch"],
                            kill_reason=scaled["reason"] if scaled["kill_switch"] else None,
                        )

                    # Update bot_state
                    async with bot_state_ref.lock:
                        bot_state_ref.fills_today += 1
                        bot_state_ref.last_fill_timestamp = fill.get("timestamp", 0)
                        bot_state_ref.scale_ratio = scaled["scale_ratio"]
                        bot_state_ref.smoothed_ratio = self.compounding_engine.smoothed_ratio
                        bot_state_ref.btc_price = fill.get("price", 0.0)

                    # Telegram / WhatsApp alerts
                    await self.telegram_alerter.send_fill(fill, scaled)
                    if self.whatsapp_alerter and self.whatsapp_alerter.enabled:
                        await self.whatsapp_alerter.send_fill(fill, scaled)

                    # ── Week 2: Mirror to MT5 ────────────────────
                    if not scaled["kill_switch"] and executor_ref:
                        mt5_result = await executor_ref.execute(scaled)
                        watcher_logger.info(
                            f"MT5 RESULT | status: {mt5_result.get('status')} | "
                            f"order_id: {mt5_result.get('mt5_order_id', 'N/A')}"
                        )

                except Exception as e:
                    from core.watcher import watcher_logger
                    watcher_logger.error(f"Error in fill callback: {e}", exc_info=True)
                    await self.telegram_alerter.send_error("FillCallback", str(e))

            # Only use single-pair watcher if multi-pair is not enabled
            if not self._multi_pair_enabled:
                self.watcher = BinanceWatcher(
                    self.config,
                    self.client,
                    self.compounding_engine,
                    self.db_manager,
                    self.telegram_alerter,
                    self.whatsapp_alerter,
                    fill_callback=_fill_callback_with_mt5,
                )

            symbol = self.config.get("binance", {}).get("symbol", "BTCUSDT")

            # Handle command-line exits immediately
            if self.args.cancel_grid:
                await self.grid_simulator.cancel_all(self.client, symbol)
                await self.client.close_connection()
                await self.telegram_alerter.stop()
                return

            if self.args.status:
                await self.compounding_engine.poll(self.client)
                ce_status = self.compounding_engine.get_status()
                sim_status = await self.grid_simulator.get_grid_status(self.client, symbol)
                print("\n--- Compounding Engine Status ---")
                for k, v in ce_status.items():
                    print(f"{k}: {v}")
                print("\n--- Grid Simulator Status ---")
                for k, v in sim_status.items():
                    print(f"{k}: {v}")
                print("\n--- Bot State ---")
                for k, v in self.bot_state.to_dict().items():
                    print(f"{k}: {v}")
                await self.client.close_connection()
                await self.telegram_alerter.stop()
                return

            # Place fresh grid if requested
            if self.args.place_grid:
                grid_cfg = self.config.get("grid", {})
                logger.info("Placing fresh grid simulator orders on testnet...")
                await self.grid_simulator.place_grid(
                    self.client,
                    symbol,
                    grid_cfg.get("initial_capital", 100.0),
                    grid_cfg.get("levels", 20),
                    grid_cfg.get("range_pct", 0.05),
                )

            # Start watcher and components
            if self._multi_pair_enabled and self.pair_watcher:
                # Week 4: Multi-pair mode — use PairWatcher and PairWorkers
                self.tasks.append(asyncio.create_task(self.pair_watcher.start()))
                for worker in self.pair_workers.values():
                    self.tasks.append(asyncio.create_task(worker.start()))
                if self.health_monitor:
                    self.tasks.append(asyncio.create_task(self.health_monitor.start()))
                logger.info("Multi-pair watcher and workers started.")
            else:
                # Single-pair fallback (Week 1-3 mode)
                await self.watcher.start()

            # ── Week 2: Start Reconciler (single-pair mode only) ─
            if not self._multi_pair_enabled:
                self.reconciler = Reconciler(
                    self.config,
                    self.bot_state,
                    self.executor,
                    self.client,
                    self.telegram_alerter,
                )
                await self.reconciler.start()

            # ── Week 2: Start Dashboard API ──────────────────────
            loop = asyncio.get_running_loop()
            start_dashboard_thread(
                self.bot_state,
                self.config,
                self,
                loop=loop,
                fill_callback=_fill_callback_with_mt5,
                pair_states=self.pair_states if self._multi_pair_enabled else None,
                pair_workers=self.pair_workers if self._multi_pair_enabled else None,
                validation_results=self.validation_results if self._multi_pair_enabled else None,
            )
            logger.info("Dashboard API thread started.")

            # ── Week 3: Regime Detector & Crash Monitor (single-pair mode only)
            if not self._multi_pair_enabled:
                self.regime_detector = RegimeDetector(
                    self.config, self.bot_state, self.telegram_alerter,
                    self.client, executor=self.executor, db_manager=self.db_manager,
                )
                self.crash_monitor = CrashMonitor(
                    self.config, self.bot_state, self.executor,
                    self.telegram_alerter, self.client,
                )

            # Run background workers
            scale_cfg = self.config.get("scaling", {})

            if not self._multi_pair_enabled:
                self.tasks.append(asyncio.create_task(
                    run_compounding_poller(self.compounding_engine, self.client, scale_cfg.get("poll_interval_seconds", 60))
                ))
                self.tasks.append(asyncio.create_task(
                    self.regime_detector.start()
                ))
                self.tasks.append(asyncio.create_task(
                    self.crash_monitor.start()
                ))

            # ── Week 2: Daily Reset Task (UTC 00:00) ─────────────
            self.tasks.append(asyncio.create_task(
                self._daily_reset_loop()
            ))

            # Register termination signals
            loop = asyncio.get_running_loop()
            for sig in (signal.SIGINT, signal.SIGTERM):
                try:
                    loop.add_signal_handler(sig, lambda s=sig: asyncio.create_task(self.shutdown(s)))
                except NotImplementedError:
                    pass

            # Windows fallback: SIGINT via signal module (works on ProactorEventLoop)
            if sys.platform == "win32":
                signal.signal(signal.SIGINT, lambda *_: asyncio.create_task(self.shutdown(signal.SIGINT)))

            if self._multi_pair_enabled:
                pair_list = ", ".join(self.pair_states.keys())
                await self.whatsapp_alerter.send(
                    f"Bot started — Week 4 multi-pair active.\nPairs: {pair_list}"
                )
            else:
                await self.telegram_alerter.send("🤖 Bot started — Week 3 full system active.")
            await self.shutdown_event.wait()

        except (KeyboardInterrupt, asyncio.CancelledError):
            logger.info("Bot execution interrupted or cancelled.")
        finally:
            if not self.shutdown_event.is_set():
                await self.shutdown()

    async def _setup_multi_pair(self) -> None:
        """Initialises all multi-pair components: validation, states, workers, watcher."""
        pairs_cfg = self.config.get("pairs", {})
        active_pairs = {
            pair: cfg
            for pair, cfg in pairs_cfg.items()
            if cfg.get("enabled", False)
        }

        if not active_pairs:
            logger.warning("Multi-pair enabled but no pairs configured — falling back to single-pair")
            self._multi_pair_enabled = False
            return

        # 1. Validate MT5 symbols
        validator = SymbolValidator(self.config)
        self.validation_results = validator.validate_all_pairs(active_pairs)
        unknown = validator.get_unknown_pairs(self.validation_results)

        if unknown:
            for pair in unknown:
                msg = (
                    f"WARNING: {pair} ({active_pairs[pair].get('mt5_symbol', '')}) "
                    f"not available on MT5. This pair will be skipped."
                )
                logger.warning(msg)
                if self.whatsapp_alerter and self.whatsapp_alerter.enabled:
                    await self.whatsapp_alerter.send(msg)

        # 2. Create per-pair state
        self.pair_states = {
            pair: PairState(
                pair=pair,
                mt5_symbol=cfg["mt5_symbol"],
                grid_capital=cfg["grid_capital"],
                mirror_enabled=cfg.get("mirror_enabled", True),
                mt5_symbol_available=self.validation_results[pair].available,
                mt5_symbol_checked=True,
            )
            for pair, cfg in active_pairs.items()
        }

        # 3. Create PairWorkers (only for available symbols)
        self.pair_workers = {}
        for pair, cfg in active_pairs.items():
            if not self.validation_results[pair].available:
                continue
            worker = PairWorker(
                pair=pair,
                pair_config=cfg,
                global_config=self.config,
                pair_state=self.pair_states[pair],
                mt5_executor=self.executor,
                telegram=self.whatsapp_alerter,
                binance_client=self.client,
                db_manager=self.db_manager,
                all_pair_states=self.pair_states,
            )
            self.pair_workers[pair] = worker

        # 4. Create PairWatcher
        fill_callbacks = {
            pair: worker.on_fill
            for pair, worker in self.pair_workers.items()
        }
        self.pair_watcher = PairWatcher(
            self.config, self.pair_states, fill_callbacks, self.client
        )

        # 5. Create HealthMonitor
        self.health_monitor = HealthMonitor(
            self.pair_workers, self.bot_state, self.whatsapp_alerter
        )

        # Print banner after setup
        self.print_banner()

        logger.info(
            f"Multi-pair setup complete — {len(self.pair_workers)} workers, "
            f"{len(unknown)} unavailable"
        )

    async def _daily_reset_loop(self) -> None:
        """Sleeps until the next UTC 00:00, then resets daily counters."""
        from datetime import datetime, timezone, timedelta
        while True:
            try:
                now = datetime.now(timezone.utc)
                tomorrow = (now + timedelta(days=1)).replace(
                    hour=0, minute=0, second=0, microsecond=0
                )
                seconds_until = (tomorrow - now).total_seconds()
                logger.info(f"Daily reset scheduled in {seconds_until:.0f}s (next UTC 00:00).")
                await asyncio.sleep(seconds_until)

                await self.bot_state.reset_daily()
                if self.executor:
                    self.executor.reset_day_open_equity()
                logger.info("Daily counters reset at UTC 00:00.")
                await self.telegram_alerter.send("🔄 Daily counters reset — new trading day.")
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in daily reset loop: {e}", exc_info=True)
                await asyncio.sleep(60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Grid Mirror Bot execution CLI.")
    parser.add_argument("--place-grid", action="store_true", help="Place fresh simulated grid on startup.")
    parser.add_argument("--cancel-grid", action="store_true", help="Cancel all open orders and exit.")
    parser.add_argument("--status", action="store_true", help="Print current status and exit.")
    parser.add_argument("--no-telegram", action="store_true", help="Disable Telegram for this run.")
    args = parser.parse_args()

    runner = BotRunner(load_config(), args)
    try:
        asyncio.run(runner.execute())
    except KeyboardInterrupt:
        pass
