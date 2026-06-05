"""
WhatsApp Alert Service for the Grid Mirror Bot.
Handles non-blocking, queue-based notifications via CallMeBot or Twilio APIs.
"""

import asyncio
import logging
import urllib.parse
from typing import Any, Dict
import aiohttp

logger = logging.getLogger("whatsapp_alerter")


class WhatsAppAlerter:
    """Queue-based, non-blocking WhatsApp alert manager."""

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initializes the WhatsApp Alerter.

        Args:
            config (dict): The configuration dictionary (including whatsapp and env secrets).
        """
        self.config = config
        wa_cfg = config.get("whatsapp", {})
        self.enabled = wa_cfg.get("enabled", False)
        self.provider = wa_cfg.get("provider", "callmebot").lower()
        self.phone_number = config.get("WHATSAPP_PHONE") or wa_cfg.get("phone_number")
        self.api_key = config.get("WHATSAPP_API_KEY")

        # Twilio specific keys
        self.account_sid = config.get("WHATSAPP_ACCOUNT_SID")
        self.auth_token = config.get("WHATSAPP_AUTH_TOKEN")
        self.twilio_from = wa_cfg.get("twilio_from_number", "whatsapp:+14155238886")

        self.queue: asyncio.Queue[str] = asyncio.Queue()
        self.worker_task: asyncio.Task[None] | None = None
        self.session: aiohttp.ClientSession | None = None

        if self.enabled and self.phone_number:
            logger.info(f"WhatsApp Alerter initialized using provider: {self.provider}")
        else:
            self.enabled = False
            logger.info("WhatsApp alerts are disabled or configurations are missing.")

    async def start(self) -> None:
        """
        Starts the background worker queue consumer.
        """
        if not self.enabled:
            return
        self.session = aiohttp.ClientSession()
        self.worker_task = asyncio.create_task(self._worker())
        logger.info("WhatsApp alert queue worker started.")

    async def stop(self) -> None:
        """
        Stops the worker task and cleans up the HTTP session.
        """
        if self.worker_task:
            self.worker_task.cancel()
            try:
                await self.worker_task
            except asyncio.CancelledError:
                pass
        if self.session:
            await self.session.close()
        logger.info("WhatsApp alert queue worker stopped.")

    async def _worker(self) -> None:
        """
        Background worker that processes and sends queued alerts.
        """
        while True:
            message = await self.queue.get()
            success = False
            for attempt in range(1, 4):
                try:
                    if self.provider == "callmebot":
                        success = await self._send_callmebot(message)
                    elif self.provider == "textmebot":
                        success = await self._send_textmebot(message)
                    elif self.provider == "twilio":
                        success = await self._send_twilio(message)
                    
                    if success:
                        break
                except Exception as e:
                    logger.warning(
                        f"WhatsApp send attempt {attempt} failed: {e}"
                    )
                await asyncio.sleep(5.0)

            if not success:
                logger.error(f"Failed to send WhatsApp alert after 3 attempts: {message[:100]}")

            self.queue.task_done()

    async def _send_callmebot(self, message: str) -> bool:
        """Sends a message using the CallMeBot API."""
        if not self.session or not self.api_key or not self.phone_number:
            return False
        
        # CallMeBot accepts messages via URL parameter
        encoded_msg = urllib.parse.quote_plus(message)
        url = (
            f"https://api.callmebot.com/whatsapp.php"
            f"?phone={self.phone_number}&text={encoded_msg}&apikey={self.api_key}"
        )
        
        async with self.session.get(url) as response:
            if response.status == 200:
                logger.debug("CallMeBot message sent successfully.")
                return True
            else:
                text = await response.text()
                logger.warning(f"CallMeBot returned status {response.status}: {text}")
                return False

    async def _send_textmebot(self, message: str) -> bool:
        """Sends a message using the TextMeBot API."""
        if not self.session or not self.api_key or not self.phone_number:
            return False
        
        # TextMeBot accepts messages via URL parameters
        encoded_msg = urllib.parse.quote_plus(message)
        url = (
            f"https://api.textmebot.com/send.php"
            f"?recipient={self.phone_number}&text={encoded_msg}&apikey={self.api_key}"
        )
        
        async with self.session.get(url) as response:
            if response.status == 200:
                logger.debug("TextMeBot message sent successfully.")
                return True
            else:
                text = await response.text()
                logger.warning(f"TextMeBot returned status {response.status}: {text}")
                return False

    async def _send_twilio(self, message: str) -> bool:
        """Sends a message using the Twilio API."""
        if not self.session or not self.account_sid or not self.auth_token or not self.phone_number:
            return False

        url = f"https://api.twilio.com/2010-04-01/Accounts/{self.account_sid}/Messages.json"
        
        # Ensure numbers are prefixed with whatsapp:
        to_number = self.phone_number
        if not to_number.startswith("whatsapp:"):
            to_number = f"whatsapp:{to_number}"

        data = {
            "From": self.twilio_from,
            "To": to_number,
            "Body": message
        }
        
        auth = aiohttp.BasicAuth(self.account_sid, self.auth_token)
        async with self.session.post(url, data=data, auth=auth) as response:
            if response.status in [200, 201]:
                logger.debug("Twilio WhatsApp message sent successfully.")
                return True
            else:
                text = await response.text()
                logger.warning(f"Twilio returned status {response.status}: {text}")
                return False

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
            "🔔 *FILL DETECTED*\n"
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
            "📊 *BOT STATUS REPORT*\n"
            f"Binance Balance: ${status.get('binance_balance', 0.0):,.2f}\n"
            f"FN Equity: ${status.get('fn_equity', 0.0):,.2f}\n"
            f"Current Scale Ratio: {status.get('scale_ratio', 0.0):.2f}x\n"
            f"Drawdown: {status.get('drawdown_pct', 0.0):.2%}\n"
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
            "⚠️ *CRASH MONITOR ALERT*\n"
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
        msg = f"❌ *ERROR* in {component}\nError Detail: {error}"
        await self.queue.put(msg)
