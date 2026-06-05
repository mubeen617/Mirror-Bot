"""
Telegram Alert Service for the Grid Mirror Bot.
Handles non-blocking, queue-based notifications via the Telegram Bot API.
"""

import asyncio
import logging
from typing import Any, Dict
from telegram import Bot

logger = logging.getLogger("telegram_alerter")


class TelegramAlerter:
    """Queue-based, non-blocking Telegram alert manager."""

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initializes the Telegram Alerter.

        Args:
            config (dict): The configuration dictionary (including telegram and env secrets).
        """
        self.config = config
        self.enabled = config.get("telegram", {}).get("enabled", False)
        self.token = config.get("TELEGRAM_BOT_TOKEN")
        self.chat_id = config.get("TELEGRAM_CHAT_ID")
        self.queue: asyncio.Queue[str] = asyncio.Queue()
        self.bot: Bot | None = None
        self.worker_task: asyncio.Task[None] | None = None

        if self.enabled and self.token and self.chat_id:
            try:
                self.bot = Bot(token=self.token)
                logger.info("Telegram Alerter initialized successfully.")
            except Exception as e:
                logger.error(f"Failed to initialize Telegram Bot: {e}", exc_info=True)
                self.enabled = False
        else:
            logger.info("Telegram alerts are disabled or credentials are missing.")
            self.enabled = False

    async def start(self) -> None:
        """
        Starts the background worker queue consumer task.
        """
        if not self.enabled:
            return
        self.worker_task = asyncio.create_task(self._worker())
        logger.info("Telegram alert queue worker started.")

    async def stop(self) -> None:
        """
        Stops the worker task and processes remaining items.
        """
        if self.worker_task:
            self.worker_task.cancel()
            try:
                await self.worker_task
            except asyncio.CancelledError:
                pass
            logger.info("Telegram alert queue worker stopped.")

    async def _worker(self) -> None:
        """
        Background worker that processes and sends queued alerts.
        """
        while True:
            message = await self.queue.get()
            success = False
            for attempt in range(1, 4):
                try:
                    if self.bot and self.chat_id:
                        await self.bot.send_message(chat_id=self.chat_id, text=message)
                        success = True
                        break
                except Exception as e:
                    logger.warning(
                        f"Telegram send attempt {attempt} failed for msg '{message[:20]}...': {e}"
                    )
                    await asyncio.sleep(5.0)

            if not success:
                logger.error(f"Failed to send Telegram alert after 3 attempts: {message[:100]}")

            self.queue.task_done()

    async def send(self, message: str) -> None:
        """
        Queues a plain text message.
        """
        if not self.enabled:
            return
        await self.queue.put(message)

    async def send_fill(self, fill: Dict[str, Any], scaled: Dict[str, Any]) -> None:
        """
        Formats and queues a fill notification.
        """
        if not self.enabled:
            return
        kill_switch_str = "ON" if scaled.get("kill_switch", False) else "OFF"
        reason_str = f" Reason: {scaled.get('reason')}" if scaled.get("kill_switch", False) else ""
        
        msg = (
            "🔔 FILL DETECTED\n"
            f"Side: {fill.get('side')}\n"
            f"Binance qty: {fill.get('qty')} {fill.get('symbol')} @ ${fill.get('price'):,.2f}\n"
            f"Scaled for FundedNext: {scaled.get('scaled_qty'):.4f} {scaled.get('symbol')}\n"
            f"Scale ratio: {scaled.get('scale_ratio'):.1f}x\n"
            f"Kill switch: {kill_switch_str}{reason_str}"
        )
        await self.queue.put(msg)

    async def send_status(self, status: Dict[str, Any]) -> None:
        """
        Formats and queues the full status.
        """
        if not self.enabled:
            return
        msg = (
            "📊 BOT STATUS REPORT\n"
            f"Binance Balance: ${status.get('binance_balance', 0.0):,.2f}\n"
            f"FN Equity: ${status.get('fn_equity', 0.0):,.2f}\n"
            f"Current Scale Ratio: {status.get('scale_ratio', 0.0):.2f}x\n"
            f"Drawdown: {status.get('drawdown_pct', 0.0):.2%} / {status.get('max_drawdown_limit_pct', 0.0):.2%}\n"
            f"Kill Switch: {'ON' if status.get('kill_switch', False) else 'OFF'}"
        )
        await self.queue.put(msg)

    async def send_crash(self, price: float, drop_pct: float) -> None:
        """
        Formats and queues a crash alert.
        """
        if not self.enabled:
            return
        msg = (
            "⚠️ CRASH MONITOR ALERT\n"
            f"Market price dropped suddenly to ${price:,.2f} ({drop_pct:.2%} drop).\n"
            "Bot has activated protective lockout."
        )
        await self.queue.put(msg)

    async def send_error(self, component: str, error: str) -> None:
        """
        Formats and queues an error alert.
        """
        if not self.enabled:
            return
        msg = f"❌ ERROR in {component}\nError Detail: {error}"
        await self.queue.put(msg)
