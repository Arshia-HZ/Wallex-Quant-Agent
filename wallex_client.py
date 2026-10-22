# wallex_client.py
"""
Wallex REST API Client with strict tick-size/step-size formatting,
market compliance guards, and stale order cancellation support.
"""

import math
import time
import uuid
import logging
import requests
import pandas as pd
from typing import Dict, List, Optional
from market_compliance import MarketComplianceGuard, normalize_symbol
from config import AgentConfig

logger = logging.getLogger("WallexAgent")


class WallexClient:
    """
    Wraps Wallex REST endpoints with tick-size and step-size formatting,
    compliance verification, and order lifecycle management.
    """

    def __init__(self, config: AgentConfig):
        self.config = config
        self.session = requests.Session()
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "WallexHackathonAgent/1.0",
        }
        if self.config.api_key:
            headers["X-API-Key"] = self.config.api_key
        self.session.headers.update(headers)
        self.market_rules: Dict[str, dict] = {}

    def load_market_rules(self) -> Dict[str, dict]:
        """
        Fetches market rules (precisions, minQty, minNotional, stepSize) from /v1/markets.
        Ensures execution stability and prevents order rejections.
        """
        try:
            resp = self.session.get(f"{self.config.base_url}/v1/markets", timeout=12)
            resp.raise_for_status()
            data = resp.json()
            symbols_data = data.get("result", {}).get("symbols", {})

            for sym, info in symbols_data.items():
                norm = normalize_symbol(sym)
                # Parse precision safely
                price_prec = int(info.get("quotePrecision", info.get("quoteAssetPrecision", 0) or 0))
                qty_prec = int(info.get("baseAssetPrecision", info.get("stepSize", 6) or 6))
                min_qty = float(info.get("minQty", 0.00000001))
                min_notional = float(info.get("minNotional", 50000.0))

                self.market_rules[norm] = {
                    "symbol": norm,
                    "price_precision": price_prec,
                    "qty_precision": qty_prec,
                    "min_qty": min_qty,
                    "min_notional": min_notional,
                    "base_asset": info.get("baseAsset", ""),
                    "quote_asset": info.get("quoteAsset", "TMN"),
                }

            logger.info(f"Loaded market rules for {len(self.market_rules)} Wallex symbols.")
            return self.market_rules
        except Exception as e:
            logger.error(f"Failed to load market rules: {e}")
            raise

    def get_symbol_rule(self, symbol: str) -> dict:
        norm = normalize_symbol(symbol)
        if norm not in self.market_rules:
            # Fallback default rules if market rules not yet populated
            return {
                "symbol": norm,
                "price_precision": 0,
                "qty_precision": 6,
                "min_qty": 0.00001,
                "min_notional": 50000.0,
            }
        return self.market_rules[norm]

    def format_price(self, symbol: str, price: float) -> str:
        rule = self.get_symbol_rule(symbol)
        prec = rule["price_precision"]
        if prec == 0:
            return str(int(round(price)))
        factor = 10 ** prec
        rounded = math.floor(price * factor) / factor
        return f"{rounded:.{prec}f}"

    def format_quantity(self, symbol: str, quantity: float) -> str:
        rule = self.get_symbol_rule(symbol)
        prec = rule["qty_precision"]
        factor = 10 ** prec
        # Floor round quantity to avoid exceeding available balance
        rounded = math.floor(quantity * factor) / factor
        return f"{rounded:.{prec}f}"

    def get_balances(self) -> Dict[str, dict]:
        """
        Retrieves user account balances.
        Returns mapping like: {'TMN': {'value': '5000000', 'locked': '0'}, ...}
        """
        try:
            resp = self.session.get(f"{self.config.base_url}/v1/account/balances", timeout=10)
            resp.raise_for_status()
            res = resp.json().get("result", {})
            return res.get("balances", {})
        except Exception as e:
            logger.error(f"Error fetching account balances: {e}")
            raise

    def get_orderbook(self, symbol: str) -> dict:
        """
        Fetches live order book depth from /v1/depth?symbol=...
        """
        norm = normalize_symbol(symbol)
        try:
            resp = self.session.get(f"{self.config.base_url}/v1/depth", params={"symbol": norm}, timeout=10)
            resp.raise_for_status()
            return resp.json().get("result", {})
        except Exception as e:
            logger.error(f"Error fetching order book for {norm}: {e}")
            return {"bid": [], "ask": []}

    def get_latest_trades(self, symbol: str) -> List[dict]:
        """
        Fetches latest trades from /v1/trades?symbol=...
        """
        norm = normalize_symbol(symbol)
        try:
            resp = self.session.get(f"{self.config.base_url}/v1/trades", params={"symbol": norm}, timeout=10)
            resp.raise_for_status()
            return resp.json().get("result", {}).get("latestTrades", [])
        except Exception as e:
            logger.warning(f"Error fetching latest trades for {norm}: {e}")
            return []

    def get_candles(
        self, symbol: str, resolution: str = "15", lookback_hours: int = 48
    ) -> pd.DataFrame:
        """
        Fetches historical OHLCV candle data via /v1/udf/history.
        """
        norm = normalize_symbol(symbol)
        now = int(time.time())
        frm = now - (lookback_hours * 3600)
        params = {"symbol": norm, "resolution": resolution, "from": frm, "to": now}
        try:
            resp = self.session.get(f"{self.config.base_url}/v1/udf/history", params=params, timeout=12)
            resp.raise_for_status()
            data = resp.json()
            if data.get("s") != "ok" or not data.get("t"):
                return pd.DataFrame()

            df = pd.DataFrame({
                "timestamp": data["t"],
                "open": [float(x) for x in data["o"]],
                "high": [float(x) for x in data["h"]],
                "low": [float(x) for x in data["l"]],
                "close": [float(x) for x in data["c"]],
                "volume": [float(x) for x in data["v"]],
            })
            return df
        except Exception as e:
            logger.error(f"Error fetching candles for {norm}: {e}")
            return pd.DataFrame()

    def get_open_orders(self, symbol: Optional[str] = None) -> List[dict]:
        """
        Retrieves active open orders from /v1/account/openOrders.
        """
        params = {}
        if symbol:
            params["symbol"] = normalize_symbol(symbol)
        try:
            resp = self.session.get(f"{self.config.base_url}/v1/account/openOrders", params=params, timeout=10)
            resp.raise_for_status()
            result = resp.json().get("result", {})
            orders = []
            if isinstance(result, list):
                orders = result
            elif isinstance(result, dict):
                # May group orders by symbol or contain list
                for val in result.values():
                    if isinstance(val, list):
                        orders.extend(val)
                    elif isinstance(val, dict):
                        orders.append(val)
            return orders
        except Exception as e:
            logger.error(f"Error fetching open orders: {e}")
            return None

    def get_order_status(self, client_id: str) -> Optional[dict]:
        """
        Queries detailed status of an order via GET /v1/account/orders/{client_id}.
        Returns dict with status, executedQty, executedPrice, etc., or None on network error.
        """
        try:
            url = f"{self.config.base_url}/v1/account/orders/{client_id}"
            resp = self.session.get(url, timeout=8)
            if resp.status_code == 200:
                return resp.json().get("result", {})
            elif resp.status_code == 404:
                return {"status": "NOT_FOUND", "active": False}
            logger.warning(f"Order status check for {client_id} returned HTTP {resp.status_code}")
            return None
        except Exception as e:
            logger.warning(f"Error checking order status for {client_id}: {e}")
            return None

    def cancel_order(self, client_id: str) -> bool:
        """
        Cancels an open order by clientOrderId via DELETE /v1/account/orders/{client_id}
        or fallback body {"clientOrderId": client_id}. Returns True if successfully cancelled.
        """
        if self.config.mode == ExecutionMode.SHADOW:
            return True
        try:
            url = f"{self.config.base_url}/v1/account/orders/{client_id}"
            resp = self.session.delete(url, timeout=10)
            if resp.status_code in (200, 204):
                return True
            # Fallback to body-based cancellation if path DELETE returns 404/405
            fallback_url = f"{self.config.base_url}/v1/account/orders"
            resp2 = self.session.delete(fallback_url, json={"clientOrderId": client_id}, timeout=10)
            return resp2.status_code in (200, 204)
        except Exception as e:
            logger.error(f"Failed to cancel order {client_id}: {e}")
            return False

    def place_spot_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        price: Optional[float] = None,
        client_id: Optional[str] = None,
    ) -> dict:
        """
        Places a spot order on Wallex with strict compliance and precision checks.
        """
        # 1. HARD REGULATORY GUARD: Verify market is allowed and uses TMN quote
        norm_symbol = MarketComplianceGuard.assert_market_allowed(symbol)

        # 2. Format precision
        formatted_qty = self.format_quantity(norm_symbol, quantity)
        if float(formatted_qty) <= 0:
            raise ValueError(f"Order quantity {quantity} formatted to 0. Cannot submit.")

        if not client_id:
            client_id = f"agy_{int(time.time())}_{uuid.uuid4().hex[:6]}"

        payload = {
            "symbol": norm_symbol,
            "type": order_type.upper(),
            "side": side.upper(),
            "quantity": formatted_qty,
            "client_id": client_id,
        }

        if order_type.upper() == "LIMIT":
            if price is None:
                raise ValueError("Price is required for LIMIT order.")
            payload["price"] = self.format_price(norm_symbol, price)

        # 3. Minimum notional check
        rule = self.get_symbol_rule(norm_symbol)
        est_price = price if price is not None else 1.0
        notional = float(formatted_qty) * est_price
        if price is not None and notional < rule["min_notional"]:
            raise ValueError(
                f"Order notional {notional:.0f} TMN is below minimum {rule['min_notional']} TMN."
            )

        resp = self.session.post(f"{self.config.base_url}/v1/account/orders", json=payload, timeout=12)
        resp.raise_for_status()
        return resp.json().get("result", {})
