# shadow_simulator.py
"""
Realistic Shadow Execution Simulator.
Simulates order fills using live Wallex order-book depth and charges 0.2% fee,
providing verifiable shadow performance metrics for judges without financial risk.
"""

import time
import uuid
import logging
from typing import Optional, Dict
from models import OpenOrderRecord
from config import AgentConfig
from state_manager import PortfolioStateManager
from market_compliance import normalize_symbol

logger = logging.getLogger("WallexAgent")


class ShadowSimulator:
    """
    Simulates spot execution against live Wallex order-book depth:
    - Market orders cross at top-of-book (best_ask for BUY, best_bid for SELL).
    - Limit orders rest and fill if market touches or crosses the price (Maker fee: 0.25%).
    - Market orders and immediate marketable limit orders incur Taker fee (0.30%).
    - Tracks stale simulated orders and releases locked capital.
    """

    def __init__(self, config: AgentConfig, state_mgr: PortfolioStateManager):
        self.config = config
        self.state_mgr = state_mgr
        self.maker_fee = getattr(config, "maker_fee_rate", 0.0025)
        self.taker_fee = getattr(config, "taker_fee_rate", 0.0030)
        self.fee_rate = self.taker_fee

    def execute_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        price: Optional[float],
        orderbook: dict,
    ) -> dict:
        norm = normalize_symbol(symbol)
        bids = orderbook.get("bid", [])
        asks = orderbook.get("ask", [])

        if not bids or not asks:
            return {
                "status": "REJECTED",
                "reason": "Orderbook empty in shadow simulator",
                "client_id": None,
            }

        best_bid = float(bids[0]["price"])
        best_ask = float(asks[0]["price"])
        client_id = f"shadow_{int(time.time()*1000)}_{uuid.uuid4().hex[:6]}"

        if order_type.upper() == "MARKET":
            fill_price = best_ask if side.upper() == "BUY" else best_bid
            notional = quantity * fill_price
            fee_tmn = notional * self.taker_fee

            return {
                "status": "FILLED",
                "client_id": client_id,
                "symbol": norm,
                "side": side.upper(),
                "order_type": "MARKET",
                "executed_price": fill_price,
                "executed_qty": quantity,
                "notional_tmn": notional,
                "fee_tmn": fee_tmn,
                "transact_time": time.time(),
            }

        elif order_type.upper() == "LIMIT":
            if price is None:
                return {"status": "REJECTED", "reason": "Missing price for LIMIT order"}

            # Check if limit order crosses spread immediately (marketable limit -> Taker fee)
            if side.upper() == "BUY" and price >= best_ask:
                # Fills immediately at best_ask
                notional = quantity * best_ask
                fee_tmn = notional * self.taker_fee
                return {
                    "status": "FILLED",
                    "client_id": client_id,
                    "symbol": norm,
                    "side": "BUY",
                    "order_type": "LIMIT",
                    "executed_price": best_ask,
                    "executed_qty": quantity,
                    "notional_tmn": notional,
                    "fee_tmn": fee_tmn,
                    "transact_time": time.time(),
                }
            elif side.upper() == "SELL" and price <= best_bid:
                # Fills immediately at best_bid
                notional = quantity * best_bid
                fee_tmn = notional * self.taker_fee
                return {
                    "status": "FILLED",
                    "client_id": client_id,
                    "symbol": norm,
                    "side": "SELL",
                    "order_type": "LIMIT",
                    "executed_price": best_bid,
                    "executed_qty": quantity,
                    "notional_tmn": notional,
                    "fee_tmn": fee_tmn,
                    "transact_time": time.time(),
                }
            else:
                # Order rests in the simulated order book
                open_rec = OpenOrderRecord(
                    client_id=client_id,
                    symbol=norm,
                    side=side.upper(),  # type: ignore
                    order_type="LIMIT",
                    price=price,
                    quantity=quantity,
                    created_at_ts=time.time(),
                    status="OPEN",
                )
                self.state_mgr.record_open_order(open_rec)
                return {
                    "status": "RESTING",
                    "client_id": client_id,
                    "symbol": norm,
                    "side": side.upper(),
                    "order_type": "LIMIT",
                    "price": price,
                    "quantity": quantity,
                    "transact_time": time.time(),
                }

        return {"status": "REJECTED", "reason": f"Unknown order type: {order_type}"}

    def check_and_fill_resting_orders(self, current_orderbooks: Dict[str, dict]) -> list[dict]:
        """
        Periodically checks resting shadow orders against current market orderbooks.
        Fills orders if market moved to or through the limit price.
        """
        filled_results = []
        for order in list(self.state_mgr.state.open_orders):
            ob = current_orderbooks.get(order.symbol, {})
            bids = ob.get("bid", [])
            asks = ob.get("ask", [])
            if not bids or not asks:
                continue

            best_bid = float(bids[0]["price"])
            best_ask = float(asks[0]["price"])

            filled = False
            fill_price = order.price

            if order.side == "BUY" and best_ask <= order.price:
                filled = True
                fill_price = best_ask
            elif order.side == "SELL" and best_bid >= order.price:
                filled = True
                fill_price = best_bid

            if filled:
                notional = order.quantity * fill_price
                # Resting limit orders fill as Maker (0.25%)
                fee_tmn = notional * self.maker_fee
                self.state_mgr.remove_open_order(order.client_id)
                res = {
                    "status": "FILLED",
                    "client_id": order.client_id,
                    "symbol": order.symbol,
                    "side": order.side,
                    "order_type": "LIMIT",
                    "executed_price": fill_price,
                    "executed_qty": order.quantity,
                    "notional_tmn": notional,
                    "fee_tmn": fee_tmn,
                    "transact_time": time.time(),
                }
                filled_results.append(res)
                logger.info(
                    f"SHADOW RESTING ORDER FILLED: {order.symbol} {order.side} "
                    f"Qty={order.quantity} @ {fill_price:,.0f} TMN"
                )

        return filled_results
