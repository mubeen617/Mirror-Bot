"""
Multi-Pair Binance WebSocket Watcher — Week 4 Core Component.
Manages a single Binance user data stream and routes fills to the
appropriate PairWorker based on symbol. Replaces the single-pair
watcher for multi-pair operation while leaving the original watcher.py untouched.
"""

import asyncio
import logging
import time
import traceback
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Callable, Dict

try:
    from binance import AsyncClient, BinanceSocketManager
except ImportError:
    AsyncClient = None  # type: ignore[assignment,misc]
    BinanceSocketManager = None  # type: ignore[assignment,misc]

from core.pair_state import PairState

# ── Logger setup ────────────────────────────────────────────────────
_logs_dir = Path(__file__).parent.parent / "logs"
_logs_dir.mkdir(parents=True, exist_ok=True)
_log_file = _logs_dir / "pair_watcher.log"

logger = logging.getLogger("pair_watcher")
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    _handler = RotatingFileHandler(_log_file, maxBytes=10 * 1024 * 1024, backupCount=5)
    _formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    _handler.setFormatter(_formatter)
    logger.addHandler(_handler)


class PairWatcher:
    """Multi-pair Binance user data stream watcher.

    A single instance manages one WebSocket connection and routes
    execution fills to per-pair callbacks based on the fill's symbol.

    Args:
        config: Full application configuration dictionary.
        pair_states: Dict of PairState instances keyed by Binance pair.
        fill_callbacks: Dict of async fill callbacks keyed by Binance pair.
        binance_client: Authenticated AsyncClient instance.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        pair_states: Dict[str, PairState],
        fill_callbacks: Dict[str, Callable],
        binance_client: AsyncClient,
    ) -> None:
        self.config = config
        self.pair_states = pair_states
        self.fill_callbacks = fill_callbacks
        self.client = binance_client
        self.bsm = BinanceSocketManager(self.client) if BinanceSocketManager else None

        self.listen_key: str | None = None
        self.keepalive_task: asyncio.Task | None = None
        self.main_task: asyncio.Task | None = None
        self.should_run = False

        self._configured_pairs = set(pair_states.keys())
        self._enabled_pairs = {
            pair for pair, state in pair_states.items()
            if state.mirror_enabled or state.mt5_symbol_available
        }
        self._unknown_alerted: set = set()

    async def start(self) -> None:
        """Starts the WebSocket stream and begins routing fills."""
        self.should_run = True
        self.main_task = asyncio.create_task(self._main_loop())
        logger.info(
            f"PairWatcher started — monitoring {len(self._configured_pairs)} pairs: "
            f"{', '.join(sorted(self._configured_pairs))}"
        )

    async def stop(self) -> None:
        """Stops the watcher and cleans up connections."""
        self.should_run = False
        if self.keepalive_task:
            self.keepalive_task.cancel()
            try:
                await self.keepalive_task
            except asyncio.CancelledError:
                pass

        if self.main_task:
            self.main_task.cancel()
            try:
                await self.main_task
            except asyncio.CancelledError:
                pass

        if self.listen_key:
            try:
                await self.client.stream_close_listen_key(self.listen_key)
            except Exception as e:
                logger.warning(f"Error closing listen key on shutdown: {e}")

        logger.info("PairWatcher stopped.")

    async def _main_loop(self) -> None:
        """Main loop with exponential backoff reconnection and polling fallback."""
        attempt = 0
        backoff = 2.0

        while self.should_run:
            try:
                self._should_reconnect = False

                try:
                    self.listen_key = await self.client.stream_get_listen_key()
                    logger.info(f"Listen key generated: {self.listen_key[:8]}...")
                except Exception as e:
                    err_msg = str(e).lower()
                    if "410" in err_msg or "gone" in err_msg:
                        logger.warning(
                            "Binance user data stream returned 410 Gone — "
                            "switching to polling fallback"
                        )
                        await self._polling_loop()
                        return
                    raise

                if self.keepalive_task:
                    self.keepalive_task.cancel()
                self.keepalive_task = asyncio.create_task(self._keepalive_loop())

                attempt = 0
                backoff = 2.0

                async with self.bsm.user_socket() as user_socket:
                    while self.should_run and not self._should_reconnect:
                        msg = await user_socket.recv()
                        if not msg:
                            continue
                        await self._handle_socket_msg(msg)

            except asyncio.CancelledError:
                break
            except Exception as e:
                attempt += 1
                tb = traceback.format_exc()
                logger.error(
                    f"PairWatcher exception (attempt {attempt}). "
                    f"Reconnecting in {backoff}s. Error: {e}\n{tb}"
                )
                await asyncio.sleep(backoff)
                backoff = min(60.0, backoff * 2.0)

    async def _polling_loop(self) -> None:
        """Polling fallback when WebSocket endpoints are retired."""
        logger.info("Initializing multi-pair polling fallback...")
        processed_ids: Dict[str, set] = {pair: set() for pair in self._configured_pairs}
        poll_interval = 10.0
        consecutive_errors = 0

        for pair in self._configured_pairs:
            try:
                initial_trades = await self.client.get_my_trades(symbol=pair, limit=20)
                for t in initial_trades:
                    processed_ids[pair].add(str(t["id"]))
            except Exception as e:
                logger.error(f"Failed to populate initial trades for {pair}: {e}")

        logger.info(f"Polling fallback initialized for {len(self._configured_pairs)} pairs")

        while self.should_run:
            try:
                for pair in self._configured_pairs:
                    trades = await self.client.get_my_trades(symbol=pair, limit=20)
                    for t in trades:
                        trade_id = str(t["id"])
                        if trade_id not in processed_ids[pair]:
                            processed_ids[pair].add(trade_id)
                            fill = {
                                "symbol": t["symbol"],
                                "side": "BUY" if t["isBuyer"] else "SELL",
                                "qty": float(t["qty"]),
                                "price": float(t["price"]),
                                "order_id": str(t["orderId"]),
                                "timestamp": int(t["time"]),
                            }
                            await self._route_fill(fill)

                    self._update_watcher_alive(pair)

                consecutive_errors = 0
            except Exception as e:
                consecutive_errors += 1
                if consecutive_errors >= 3:
                    poll_interval = min(60.0, poll_interval * 2)
                logger.error(f"Error in polling fallback: {e}", exc_info=True)

            await asyncio.sleep(poll_interval)

    async def _keepalive_loop(self) -> None:
        """Refreshes the listen key every 30 minutes."""
        fails = 0
        while self.should_run:
            await asyncio.sleep(1800)
            if not self.listen_key:
                continue

            try:
                await self.client.stream_keepalive(self.listen_key)
                fails = 0
                logger.info("Listen key refreshed successfully.")
            except Exception as e:
                fails += 1
                logger.error(f"Failed to refresh listen key (attempt {fails}/3): {e}")
                if fails >= 3:
                    self._should_reconnect = True
                    break

    async def _handle_socket_msg(self, msg: Dict[str, Any]) -> None:
        """Parses WebSocket messages and routes fills to pair workers."""
        event_type = msg.get("e")
        if event_type == "executionReport":
            status = msg.get("X")
            if status == "FILLED":
                symbol = msg.get("s", "")
                side = msg.get("S", "")
                qty = float(msg.get("l", 0.0)) or float(msg.get("q", 0.0))
                price = float(msg.get("L", 0.0)) or float(msg.get("p", 0.0))
                order_id = str(msg.get("i", ""))
                timestamp = int(msg.get("T", int(time.time() * 1000)))

                fill = {
                    "symbol": symbol,
                    "side": side,
                    "qty": qty,
                    "price": price,
                    "order_id": order_id,
                    "timestamp": timestamp,
                }

                await self._route_fill(fill)

                self._update_watcher_alive(symbol)

    async def _route_fill(self, fill: Dict[str, Any]) -> None:
        """Routes a fill to the correct pair worker callback."""
        symbol = fill["symbol"]

        if symbol in self.fill_callbacks:
            pair_state = self.pair_states.get(symbol)
            if pair_state and not pair_state.mirror_enabled and not pair_state.mt5_symbol_available:
                logger.debug(f"Fill received for disabled pair {symbol} — skipping")
                return

            logger.info(
                f"FILL ROUTED | {symbol} | {fill['side']} | "
                f"qty={fill['qty']} | price={fill['price']} | "
                f"order_id={fill['order_id']}"
            )
            asyncio.create_task(self.fill_callbacks[symbol](fill))

        elif symbol in self._configured_pairs:
            logger.debug(f"Fill received for disabled pair {symbol} — skipping")
        else:
            logger.debug(f"Fill received for unconfigured pair {symbol} — ignoring")

    def _update_watcher_alive(self, pair: str) -> None:
        """Updates the watcher_alive timestamp for a pair."""
        if pair in self.pair_states:
            state = self.pair_states[pair]
            state.watcher_alive = True
            state.watcher_last_seen = time.time()
        for state in self.pair_states.values():
            state.watcher_alive = True
            state.watcher_last_seen = time.time()
