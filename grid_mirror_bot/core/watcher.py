"""
Binance User Data WebSocket Watcher.
Connects to Binance testnet user data stream, manages listen keys,
implements backoff reconnection, and triggers callbacks on filled orders.
"""

import asyncio
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import time
import traceback
from typing import Any, Callable, Dict
from binance import AsyncClient, BinanceSocketManager

# Setup watcher logger
logs_dir = Path(__file__).parent.parent / "logs"
logs_dir.mkdir(parents=True, exist_ok=True)
log_file = logs_dir / "watcher.log"

watcher_logger = logging.getLogger("watcher")
watcher_logger.setLevel(logging.INFO)
# Avoid adding duplicate handlers in tests or re-imports
if not watcher_logger.handlers:
    handler = RotatingFileHandler(log_file, maxBytes=10 * 1024 * 1024, backupCount=5)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    handler.setFormatter(formatter)
    watcher_logger.addHandler(handler)


class BinanceWatcher:
    """Manages Binance User Data WebSocket streams and routes execution fill events."""

    def __init__(
        self,
        config: Dict[str, Any],
        client: AsyncClient,
        compounding_engine: Any,
        db_manager: Any,
        telegram_alerter: Any,
        whatsapp_alerter: Any = None,
        fill_callback: Callable[[Dict[str, Any]], Any] | None = None,
    ) -> None:
        """
        Initializes the BinanceWatcher.
        """
        self.config = config
        self.client = client
        self.compounding_engine = compounding_engine
        self.db_manager = db_manager
        self.telegram_alerter = telegram_alerter
        self.whatsapp_alerter = whatsapp_alerter
        self.bsm = BinanceSocketManager(self.client)
        self.fill_callback = fill_callback or self.default_fill_callback

        self.listen_key: str | None = None
        self.keepalive_task: asyncio.Task[None] | None = None
        self.main_task: asyncio.Task[None] | None = None
        self.should_run = False
        self.should_reconnect = False

    async def start(self) -> None:
        """
        Starts the WebSocket watcher connection and listeners.
        """
        self.should_run = True
        self.main_task = asyncio.create_task(self._main_loop())
        watcher_logger.info("Binance user data watcher started.")

    async def stop(self) -> None:
        """
        Stops the watcher connection and cleans up tasks.
        """
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

        # Close listen key if possible
        if self.listen_key:
            try:
                await self.client.stream_close_listen_key(self.listen_key)
            except Exception as e:
                watcher_logger.warning(f"Error closing listen key on shutdown: {e}")

        watcher_logger.info("Binance user data watcher stopped.")

    async def _main_loop(self) -> None:
        """
        Main execution loop containing backoff connection logic.
        Falls back to secure polling if endpoints are retired (HTTP 410 Gone).
        """
        attempt = 0
        backoff = 2.0

        while self.should_run:
            try:
                self.should_reconnect = False
                
                # Fetch listen key
                try:
                    self.listen_key = await self.client.stream_get_listen_key()
                    watcher_logger.info(f"Listen key generated: {self.listen_key}")
                except Exception as e:
                    # Check if endpoint is retired (HTTP 410 Gone)
                    err_msg = str(e).lower()
                    if "410" in err_msg or "gone" in err_msg:
                        watcher_logger.warning(
                            "Binance user data stream REST API returned 410 Gone (Legacy listenKey retired by exchange). "
                            "Switching automatically to secure Polling Fallback stream..."
                        )
                        # Spawn the polling loop and run it cooperatively
                        await self._polling_loop()
                        return
                    else:
                        raise e

                # Start keepalive loop
                if self.keepalive_task:
                    self.keepalive_task.cancel()
                self.keepalive_task = asyncio.create_task(self._keepalive_loop())

                # Reset backoff counters upon successful connection
                attempt = 0
                backoff = 2.0

                # Connect to user socket
                async with self.bsm.user_socket() as user_socket:
                    while self.should_run and not self.should_reconnect:
                        msg = await user_socket.recv()
                        if not msg:
                            continue
                        await self._handle_socket_msg(msg)

            except asyncio.CancelledError:
                break
            except Exception as e:
                attempt += 1
                tb = traceback.format_exc()
                watcher_logger.error(
                    f"Watcher exception (attempt {attempt}). Reconnecting in {backoff}s. Error: {e}\n{tb}"
                )
                watcher_logger.warning(f"Reconnecting attempt {attempt} in {backoff}s.")
                
                # Sleep with backoff
                await asyncio.sleep(backoff)
                backoff = min(60.0, backoff * 2.0)

    async def _polling_loop(self) -> None:
        """
        Secure, authenticated Polling Fallback loop for Spot account updates.
        Triggered when Binance retires legacy User Data Stream REST endpoints (HTTP 410).
        """
        watcher_logger.info("Initializing Polling Fallback stream...")
        processed_ids = set()
        symbol = self.config.get("binance", {}).get("symbol", "BTCUSDT")
        poll_interval = 10.0
        consecutive_errors = 0

        # Populate initially to avoid processing old historical trades
        try:
            initial_trades = await self.client.get_my_trades(symbol=symbol, limit=20)
            for t in initial_trades:
                processed_ids.add(str(t["id"]))
            watcher_logger.info(f"Polling Fallback populated with {len(processed_ids)} historical trades.")
        except Exception as e:
            watcher_logger.error(f"Failed to populate initial trades in polling fallback: {e}", exc_info=True)

        while self.should_run:
            try:
                trades = await self.client.get_my_trades(symbol=symbol, limit=20)
                consecutive_errors = 0
                for t in trades:
                    trade_id = str(t["id"])
                    if trade_id not in processed_ids:
                        processed_ids.add(trade_id)

                        # Format as execution report fill
                        fill = {
                            "symbol": t["symbol"],
                            "side": "BUY" if t["isBuyer"] else "SELL",
                            "qty": float(t["qty"]),
                            "price": float(t["price"]),
                            "order_id": str(t["orderId"]),
                            "timestamp": int(t["time"]),
                        }

                        watcher_logger.info(
                            f"POLLING DETECTED FILL | ID: {trade_id} | side: {fill['side']} | "
                            f"qty: {fill['qty']} | price: {fill['price']} | order_id: {fill['order_id']}"
                        )

                        # Trigger fill callback
                        asyncio.create_task(self.fill_callback(fill))
            except Exception as e:
                consecutive_errors += 1
                if consecutive_errors >= 3:
                    poll_interval = min(60.0, poll_interval * 2)
                    watcher_logger.warning(
                        f"Rate limit backoff — polling interval increased to {poll_interval}s"
                    )
                watcher_logger.error(f"Error in Polling Fallback stream: {e}", exc_info=True)

            await asyncio.sleep(poll_interval)

    async def _keepalive_loop(self) -> None:
        """
        Periodically pings the listen key to prevent expiry.
        """
        fails = 0
        while self.should_run:
            await asyncio.sleep(1800)  # Refresh every 30 minutes
            if not self.listen_key:
                continue

            try:
                await self.client.stream_keepalive(self.listen_key)
                fails = 0
                watcher_logger.info("Listen key refreshed successfully.")
            except Exception as e:
                fails += 1
                watcher_logger.error(
                    f"Failed to refresh listen key (attempt {fails}/3): {e}", exc_info=True
                )
                if fails >= 3:
                    await self.telegram_alerter.send_error(
                        "Watcher",
                        f"Listen key refresh failed 3 times. Triggering reconnect... Error: {e}",
                    )
                    self.should_reconnect = True
                    break

    async def _handle_socket_msg(self, msg: Dict[str, Any]) -> None:
        """
        Parses socket messages and routes filled executions.
        """
        event_type = msg.get("e")
        if event_type == "executionReport":
            status = msg.get("X")
            if status == "FILLED":
                # Extract confirmed fill
                symbol = msg.get("s", "")
                side = msg.get("S", "")
                qty = float(msg.get("l", 0.0)) if float(msg.get("l", 0.0)) > 0.0 else float(msg.get("q", 0.0))
                price = float(msg.get("L", 0.0)) if float(msg.get("L", 0.0)) > 0.0 else float(msg.get("p", 0.0))
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
                
                # Log fill
                watcher_logger.info(
                    f"FILL DETECTED | timestamp: {timestamp} | side: {side} | qty: {qty} | price: {price} | order_id: {order_id}"
                )

                # Process callback asynchronously
                asyncio.create_task(self.fill_callback(fill))

    async def default_fill_callback(self, fill: Dict[str, Any]) -> None:
        """
        Default fill callback implementation. Passes the fill to compounding engine,
        saves to database, writes rotating logs, and pushes Telegram/WhatsApp alerts.
        """
        try:
            # Process fill using compounding engine
            scaled = self.compounding_engine.process_fill(fill)

            # Log scaled details
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

            # Send Telegram alerts
            await self.telegram_alerter.send_fill(fill, scaled)

            # Send WhatsApp alerts
            if self.whatsapp_alerter and self.whatsapp_alerter.enabled:
                await self.whatsapp_alerter.send_fill(fill, scaled)

        except Exception as e:
            watcher_logger.error(f"Error in watcher fill callback: {e}", exc_info=True)
            await self.telegram_alerter.send_error("WatcherCallback", str(e))
            if self.whatsapp_alerter and self.whatsapp_alerter.enabled:
                await self.whatsapp_alerter.send_error("WatcherCallback", str(e))
