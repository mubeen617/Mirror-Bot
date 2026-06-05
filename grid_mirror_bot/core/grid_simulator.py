"""
Grid Simulator for the Binance Spot Testnet.
Calculates geometrically spaced levels, applies exchange filters, and handles order placements and cancellations.
"""

import asyncio
import logging
import math
import time
from typing import Any, Dict, List, Tuple

logger = logging.getLogger("grid_simulator")


class GridSimulator:
    """Calculates geometric grid levels and executes mock grid placements on the Binance Testnet."""

    @staticmethod
    def _get_precision(step: float) -> int:
        """Helper to get decimal precision from a step size."""
        if step >= 1.0:
            return 0
        return int(round(-math.log10(step)))

    @classmethod
    def _round_value(cls, value: float, step: float) -> float:
        """Rounds value to a given step size using mathematical precision."""
        precision = cls._get_precision(step)
        factor = round(value / step)
        return round(factor * step, precision)

    @classmethod
    async def _fetch_filters(cls, client: Any, symbol: str) -> Tuple[float, float, float, float]:
        """Fetches tick size and step size from symbol exchange info."""
        info = await client.get_symbol_info(symbol)
        tick_size = 0.01
        step_size = 0.01
        min_qty = 0.0
        min_notional = 0.0

        for filt in info.get("filters", []):
            if filt["filterType"] == "PRICE_FILTER":
                tick_size = float(filt["tickSize"])
            elif filt["filterType"] == "LOT_SIZE":
                step_size = float(filt["stepSize"])
                min_qty = float(filt["minQty"])
            elif filt["filterType"] == "NOTIONAL":
                min_notional = float(filt["minNotional"])

        return tick_size, step_size, min_qty, min_notional

    async def place_grid(
        self, client: Any, symbol: str, capital: float, levels: int, range_pct: float
    ) -> List[Dict[str, Any]]:
        """
        Places geometric grid limit orders on the Binance spot testnet.

        Args:
            client (AsyncClient): Binance client instance.
            symbol (str): Trading pair symbol (e.g. BTCUSDT).
            capital (float): Capital allocated to the grid.
            levels (int): Number of grid levels.
            range_pct (float): Spread range as percentage (e.g. 0.05 for 5%).

        Returns:
            list: List of dictionaries of placed orders.
        """
        # Fetch tickers and exchange filters
        ticker = await client.get_symbol_ticker(symbol=symbol)
        center = float(ticker["price"])
        tick_size, step_size, min_qty, min_notional = await self._fetch_filters(client, symbol)

        # Calculate geometric spacing
        p_min = center * (1.0 - range_pct / 2.0)
        p_max = center * (1.0 + range_pct / 2.0)
        r = (p_max / p_min) ** (1.0 / (levels - 1))
        level_prices = [p_min * (r**i) for i in range(levels)]

        capital_per_level = capital / levels
        placed_orders = []

        for price in level_prices:
            # Round price to tick size
            rounded_price = self._round_value(price, tick_size)
            side = "BUY" if rounded_price < center else "SELL"

            # Quantity and filter checks
            raw_qty = capital_per_level / rounded_price
            qty = max(min_qty, self._round_value(raw_qty, step_size))

            # Notional check
            if qty * rounded_price < min_notional:
                qty = self._round_value(min_notional / rounded_price + step_size, step_size)

            try:
                if side == "BUY":
                    order = await client.order_limit_buy(
                        symbol=symbol, quantity=qty, price=rounded_price
                    )
                else:
                    order = await client.order_limit_sell(
                        symbol=symbol, quantity=qty, price=rounded_price
                    )

                placed_orders.append(
                    {
                        "orderId": order["orderId"],
                        "side": side,
                        "qty": qty,
                        "price": rounded_price,
                    }
                )
                logger.info(f"Placed {side} order: ID {order['orderId']} | {qty} @ ${rounded_price}")
            except Exception as e:
                logger.error(f"Failed to place {side} grid order at ${rounded_price}: {e}")

        return placed_orders

    async def cancel_all(self, client: Any, symbol: str) -> int:
        """
        Cancels all open orders on the symbol.

        Args:
            client: Binance client.
            symbol: Trading symbol.

        Returns:
            int: Number of orders cancelled.
        """
        try:
            open_orders = await client.get_open_orders(symbol=symbol)
            cancel_count = 0
            for order in open_orders:
                await client.cancel_order(symbol=symbol, orderId=order["orderId"])
                cancel_count += 1
            logger.info(f"Cancelled {cancel_count} open orders on {symbol}.")
            return cancel_count
        except Exception as e:
            logger.error(f"Error cancelling open orders on {symbol}: {e}")
            return 0

    async def get_grid_status(self, client: Any, symbol: str) -> Dict[str, Any]:
        """
        Computes the current status of the grid simulator.

        Args:
            client: Binance client.
            symbol: Trading symbol.

        Returns:
            dict: Summary statistics of the active grid.
        """
        try:
            open_orders = await client.get_open_orders(symbol=symbol)
            my_trades = await client.get_my_trades(symbol=symbol, limit=100)

            now_ms = int(time.time() * 1000)
            twenty_four_hours_ago = now_ms - (24 * 60 * 60 * 1000)

            filled_today = sum(
                1 for t in my_trades if int(t["time"]) >= twenty_four_hours_ago
            )
            total_fills = len(my_trades)

            # Get high and low from open orders if available
            prices = [float(o["price"]) for o in open_orders]
            low_p = min(prices) if prices else 0.0
            high_p = max(prices) if prices else 0.0

            # Calculate capital deployed based on active buy orders
            capital_deployed = sum(
                float(o["price"]) * float(o["origQty"])
                for o in open_orders
                if o["side"] == "BUY"
            )

            return {
                "open_orders": len(open_orders),
                "filled_today": filled_today,
                "total_fills": total_fills,
                "grid_range": {"low": low_p, "high": high_p},
                "capital_deployed": capital_deployed,
            }
        except Exception as e:
            logger.error(f"Failed to fetch grid status: {e}", exc_info=True)
            return {
                "open_orders": 0,
                "filled_today": 0,
                "total_fills": 0,
                "grid_range": {"low": 0.0, "high": 0.0},
                "capital_deployed": 0.0,
            }
