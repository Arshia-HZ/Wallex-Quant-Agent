# main.py
"""
Wallex Agentic AI Spot Trading Engine — Unified Main Runner (v3).

Changes vs v2:
 1. Mode-tagged state file: a state file written by another mode is backed up, never reused
    (prevents SHADOW positions leaking into LIVE). Live baseline equity = real wallet equity.
 2. All exits are MARKET orders (a resting limit sell was previously booked as filled).
 3. Two-speed loop: fast stop-loss loop on the order book only (every few seconds) +
    slower signal loop. Stops no longer wait behind candle downloads or LLM calls.
 4. Parallel market-data fetch + candle cache + 429/5xx retry.
 5. Stop/take-profit placement is clamped (min 0.8% stop floor; LLM cannot set tighter stops).
 6. Market entries by default (maker/taker gap is only 0.05%; passive bids = adverse selection).
 7. Re-entry cooldown, loss-streak pause, time stop, BTC regime filter, no entries near the end.
 8. Robust flatten at contest end (retries, releases shadow locks, final summary).
 9. Live reconciliation: phantom positions removed, orphan balances flagged.
10. Honest audit log: decision_source + LLM latency on every record. Accurate net PnL
    (actual cash cost incl. entry fee).
11. `--smoke-test` places one minimum-size live BUY+SELL to verify API assumptions.
"""

import os
import sys
import time
import json
import uuid
import math
import signal
import logging
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
import requests
import pandas as pd

from config import AgentConfig, ExecutionMode
from market_compliance import MarketComplianceGuard, normalize_symbol
from quant_engine import QuantFeatureEngine
from llm_strategist import CostEffectiveLLMStrategist, compute_sl_tp
from models import AIDecisionOutput

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("agent_runtime.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("WallexAgent")


def _order_cid(order: dict) -> Optional[str]:
    """Different Wallex endpoints name the client id differently; accept the common spellings."""
    return order.get("clientOrderId") or order.get("client_id") or order.get("clientId")


class WallexClient:
    def __init__(self, config: AgentConfig):
        self.config = config
        self.session = requests.Session()
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "WallexHackathonAgent/3.0",
        }
        if self.config.api_key:
            headers["X-API-Key"] = self.config.api_key
        self.session.headers.update(headers)
        self.market_rules: Dict[str, dict] = {}
        self._candle_cache: Dict[str, Tuple[float, pd.DataFrame]] = {}

    # ------------------------------------------------------------------ HTTP helper
    def _get(self, path: str, params: Optional[dict] = None, timeout: int = 10, retries: int = 2):
        """GET with small retry/backoff for 429 / 5xx / network errors. 4xx raises immediately."""
        url = f"{self.config.base_url}{path}"
        last_exc = None
        last_resp = None
        for attempt in range(retries + 1):
            try:
                resp = self.session.get(url, params=params, timeout=timeout)
            except requests.RequestException as e:
                last_exc = e
                time.sleep(0.4 * (attempt + 1))
                continue
            if resp.status_code == 429 or resp.status_code >= 500:
                last_resp = resp
                time.sleep(0.6 * (attempt + 1))
                continue
            resp.raise_for_status()
            return resp
        if last_resp is not None:
            last_resp.raise_for_status()
            return last_resp
        raise last_exc  # type: ignore[misc]

    # ------------------------------------------------------------------ market rules
    def load_market_rules(self) -> List[str]:
        """Fetches market precision rules and enforces the 100,000 TMN minimum notional floor."""
        resp = self._get("/v1/markets", timeout=12)
        symbols_data = resp.json().get("result", {}).get("symbols", {})

        allowed_active = MarketComplianceGuard.filter_active_wallex_markets(
            symbols_data, top_only=True
        )
        self.market_rules = {}
        for sym in allowed_active:
            info = symbols_data.get(sym, {})
            price_prec = int(info.get("quotePrecision", info.get("quoteAssetPrecision", 0) or 0))
            qty_prec = int(info.get("baseAssetPrecision", info.get("stepSize", 6) or 6))
            min_qty = float(info.get("minQty", 0.00000001))
            # Enforce at least 100,000 TMN even if the API metadata reports 50,000 TMN
            api_min_notional = float(info.get("minNotional", 100_000.0) or 100_000.0)
            min_notional = max(api_min_notional, self.config.exchange_min_notional_tmn)

            self.market_rules[sym] = {
                "base_asset": info.get("baseAsset", sym.replace("TMN", "")),
                "price_precision": price_prec,
                "qty_precision": qty_prec,
                "min_qty": min_qty,
                "min_notional": min_notional,
            }
        logger.info(
            f"Loaded {len(self.market_rules)} compliant TMN markets "
            f"(Min Order Floor: {self.config.exchange_min_notional_tmn:,.0f} TMN)"
        )
        for sym, r in self.market_rules.items():
            logger.info(
                f"  RULES {sym}: price_prec={r['price_precision']} qty_prec={r['qty_precision']} "
                f"min_qty={r['min_qty']} min_notional={r['min_notional']:,.0f}"
            )
        return list(self.market_rules.keys())

    def format_price(self, symbol: str, price: float) -> str:
        prec = self.market_rules.get(symbol, {}).get("price_precision", 0)
        if prec == 0:
            return str(int(round(price)))
        factor = 10 ** prec
        rounded = math.floor(price * factor) / factor
        return f"{rounded:.{prec}f}"

    def format_quantity(self, symbol: str, quantity: float) -> str:
        prec = self.market_rules.get(symbol, {}).get("qty_precision", 6)
        factor = 10 ** prec
        rounded = math.floor(quantity * factor) / factor
        return f"{rounded:.{prec}f}" if prec > 0 else str(int(rounded))

    # ------------------------------------------------------------------ account
    def get_balances(self) -> Dict[str, dict]:
        resp = self._get("/v1/account/balances", timeout=10)
        return resp.json().get("result", {}).get("balances", {})

    def get_free_asset_balance(self, symbol: str) -> float:
        """Returns actual unlocked base coin balance in live Wallex wallet."""
        base_asset = self.market_rules.get(symbol, {}).get("base_asset", symbol.replace("TMN", ""))
        balances = self.get_balances()
        b_info = balances.get(base_asset, {})
        val = float(b_info.get("value", 0.0) or 0.0)
        locked = float(b_info.get("locked", 0.0) or 0.0)
        return max(0.0, val - locked)

    # ------------------------------------------------------------------ market data
    def get_orderbook(self, symbol: str) -> dict:
        norm = normalize_symbol(symbol)
        try:
            resp = self._get("/v1/depth", params={"symbol": norm}, timeout=10)
            return resp.json().get("result", {})
        except Exception as e:
            logger.warning(f"Error fetching depth for {norm}: {e}")
            return {"bid": [], "ask": []}

    def get_latest_trades(self, symbol: str) -> List[dict]:
        norm = normalize_symbol(symbol)
        try:
            resp = self._get("/v1/trades", params={"symbol": norm}, timeout=10)
            return resp.json().get("result", {}).get("latestTrades", [])
        except Exception as e:
            logger.warning(f"Error fetching latest trades for {norm}: {e}")
            return []

    def get_candles(
        self, symbol: str, resolution: str = "15", lookback_hours: int = 72
    ) -> pd.DataFrame:
        """
        Fetches historical OHLCV candle data via /v1/udf/history.
        Automatically resamples 1m data to true 15m or 60m candles if Wallex returns 1m resolution.
        """
        norm = normalize_symbol(symbol)
        now = int(time.time())
        frm = now - (lookback_hours * 3600)
        params = {"symbol": norm, "resolution": resolution, "from": frm, "to": now}
        try:
            resp = self._get("/v1/udf/history", params=params, timeout=25)
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

            # Wallex UDF API returns 1-minute candles even when resolution="15" is requested.
            # Resample into proper 15-minute or 60-minute candles for accurate multi-candle quant indicators.
            if len(df) > 2 and (df["timestamp"].iloc[1] - df["timestamp"].iloc[0]) <= 120 and resolution in ("15", "60"):
                freq = "15min" if resolution == "15" else "60min"
                df["dt"] = pd.to_datetime(df["timestamp"], unit="s")
                df = df.set_index("dt").resample(freq).agg({
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last",
                    "volume": "sum",
                    "timestamp": "first",
                }).dropna().reset_index(drop=True)

            return df
        except Exception as e:
            logger.warning(f"Error fetching candles for {norm}: {e}")
            return pd.DataFrame()

    def get_candles_cached(
        self, symbol: str, resolution: str = "15", lookback_hours: int = 72
    ) -> pd.DataFrame:
        """Candles refreshed at most every candle_cache_sec; stale copy (<10 min) used if a refresh fails."""
        norm = normalize_symbol(symbol)
        now = time.time()
        hit = self._candle_cache.get(norm)
        if hit and (now - hit[0]) < self.config.candle_cache_sec and not hit[1].empty:
            return hit[1]
        df = self.get_candles(norm, resolution=resolution, lookback_hours=lookback_hours)
        if not df.empty:
            self._candle_cache[norm] = (now, df)
            return df
        if hit and (now - hit[0]) < 600:
            return hit[1]
        return df

    # ------------------------------------------------------------------ orders
    def get_open_orders(self, symbol: Optional[str] = None) -> Optional[List[dict]]:
        """Returns None on network error so caller never confuses an API error with filled orders."""
        if self.config.mode == ExecutionMode.SHADOW:
            return []
        params = {"symbol": normalize_symbol(symbol)} if symbol else {}
        try:
            resp = self._get("/v1/account/openOrders", params=params, timeout=10)
            result = resp.json().get("result", {})
            if isinstance(result, list):
                return result
            if isinstance(result, dict):
                if "orders" in result and isinstance(result["orders"], list):
                    return result["orders"]
                all_orders = []
                for v in result.values():
                    if isinstance(v, list):
                        all_orders.extend(v)
                    elif isinstance(v, dict):
                        all_orders.append(v)
                return all_orders
            return []
        except Exception as e:
            logger.warning(f"Error fetching open orders (skipping state promotion): {e}")
            return None

    def get_order_by_client_id(self, client_order_id: str) -> Optional[dict]:
        """Queries exact order status via /v1/account/orders/{client_id}."""
        if self.config.mode == ExecutionMode.SHADOW:
            return None
        try:
            resp = self._get(f"/v1/account/orders/{client_order_id}", timeout=10, retries=1)
            if resp.status_code == 200:
                return resp.json().get("result", {})
        except Exception as e:
            logger.debug(f"Could not inspect client_id {client_order_id}: {e}")
        return None

    def cancel_order(self, client_order_id: str) -> bool:
        if self.config.mode == ExecutionMode.SHADOW:
            return True
        try:
            url = f"{self.config.base_url}/v1/account/orders/{client_order_id}"
            resp = self.session.delete(url, timeout=10)
            if resp.status_code in (200, 204):
                return True
            resp2 = self.session.delete(
                f"{self.config.base_url}/v1/account/orders",
                json={"clientOrderId": client_order_id},
                timeout=10,
            )
            return resp2.status_code in (200, 204)
        except Exception as e:
            logger.warning(f"Failed to cancel order {client_order_id}: {e}")
            return False

    def place_spot_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        price: Optional[float] = None,
        reference_price: Optional[float] = None,
    ) -> dict:
        """
        Places a spot order after verifying:
        1. Symbol is not in the 326 forbidden markets.
        2. Post-rounding notional value (formatted_qty * price) is >= 100,000 TMN.
        Always returns a dict containing 'clientOrderId' (the id we sent) so callers can track it.
        """
        norm_symbol = MarketComplianceGuard.assert_market_allowed(symbol)
        rule = self.market_rules.get(norm_symbol, {})
        min_notional = max(
            rule.get("min_notional", 100_000.0), self.config.exchange_min_notional_tmn
        )

        formatted_qty = self.format_quantity(norm_symbol, quantity)
        final_qty_float = float(formatted_qty)
        if final_qty_float <= 0:
            raise ValueError(f"Formatted quantity for {norm_symbol} is zero ({quantity})")

        # Determine effective price to validate the 100,000 TMN minimum notional requirement
        check_price = price or reference_price
        if check_price is not None:
            notional_tmn = final_qty_float * float(check_price)
            if notional_tmn < min_notional:
                raise ValueError(
                    f"LOW_AMOUNT_GUARD: {norm_symbol} {side} order value after rounding is "
                    f"{notional_tmn:,.0f} TMN, which is below Wallex's {min_notional:,.0f} TMN minimum!"
                )

        client_id = f"wlx_{int(time.time()*1000)}_{uuid.uuid4().hex[:5]}"
        payload = {
            "symbol": norm_symbol,
            "type": order_type.upper(),
            "side": side.upper(),
            "quantity": formatted_qty,
            "client_id": client_id,
        }
        if order_type.upper() == "LIMIT":
            if price is None:
                raise ValueError("Price is required for LIMIT order")
            payload["price"] = self.format_price(norm_symbol, price)

        if self.config.mode == ExecutionMode.SHADOW:
            return {
                "status": "SHADOW_SUBMITTED",
                "clientOrderId": client_id,
                "symbol": norm_symbol,
                "side": side.upper(),
                "type": order_type.upper(),
                "origQty": formatted_qty,
                "price": payload.get("price", str(check_price or 0.0)),
            }

        resp = self.session.post(
            f"{self.config.base_url}/v1/account/orders", json=payload, timeout=12
        )
        resp.raise_for_status()
        result = resp.json().get("result", {})
        if not isinstance(result, dict):
            result = {"raw": result}
        result["_sent_client_id"] = client_id
        if not result.get("clientOrderId"):
            result["clientOrderId"] = client_id
        return result


@dataclass
class OpenPosition:
    symbol: str
    entry_price: float
    quantity: float
    stop_loss_price: float
    take_profit_price: float
    highest_price: float
    opened_at: str
    client_id: str
    trailing_active: bool = False
    cost_tmn: float = 0.0          # actual cash spent on entry (includes the entry fee)
    entry_fee_rate: float = 0.0030


def _age_sec(iso_ts: Optional[str]) -> float:
    if not iso_ts:
        return 0.0
    try:
        return max(0.0, (datetime.now(timezone.utc) - datetime.fromisoformat(iso_ts)).total_seconds())
    except Exception:
        return 0.0


class PortfolioAndRiskManager:
    def __init__(self, config: AgentConfig):
        self.config = config
        self.shadow_available_tmn: float = config.initial_capital_tmn
        self.peak_equity_tmn: float = config.initial_capital_tmn
        self.baseline_equity_tmn: float = config.initial_capital_tmn
        # In LIVE the baseline is taken from the real wallet on the first successful equity read.
        self.baseline_set: bool = config.mode != ExecutionMode.LIVE
        self.realized_pnl_tmn: float = 0.0
        self.total_trades_count: int = 0
        self.winning_trades_count: int = 0
        self.losing_trades_count: int = 0
        self.positions: Dict[str, OpenPosition] = {}
        self.pending_buy_orders: Dict[str, dict] = {}
        self.last_exit_ts: Dict[str, float] = {}
        self.consecutive_losses: int = 0
        self.entries_paused_until_ts: float = 0.0
        self._orphan_warned: set = set()
        if os.path.exists(self.config.state_file):
            self.load_state()
        else:
            self.save_state()

    # ------------------------------------------------------------------ persistence
    def save_state(self):
        data = {
            "mode": self.config.mode.value,
            "shadow_available_tmn": self.shadow_available_tmn,
            "peak_equity_tmn": self.peak_equity_tmn,
            "baseline_equity_tmn": self.baseline_equity_tmn,
            "baseline_set": self.baseline_set,
            "realized_pnl_tmn": self.realized_pnl_tmn,
            "total_trades_count": self.total_trades_count,
            "winning_trades_count": self.winning_trades_count,
            "losing_trades_count": self.losing_trades_count,
            "positions": {sym: asdict(pos) for sym, pos in self.positions.items()},
            "pending_buy_orders": self.pending_buy_orders,
            "last_exit_ts": self.last_exit_ts,
            "consecutive_losses": self.consecutive_losses,
            "entries_paused_until_ts": self.entries_paused_until_ts,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        temp_file = f"{self.config.state_file}.tmp"
        try:
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            os.replace(temp_file, self.config.state_file)
        except Exception as e:
            logger.error(f"Error saving state to {self.config.state_file}: {e}")

    def load_state(self):
        if not os.path.exists(self.config.state_file):
            return
        try:
            with open(self.config.state_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Old state files have no "mode" key; they were written by the SHADOW runs.
            saved_mode = str(data.get("mode") or "SHADOW").upper()
            if saved_mode != self.config.mode.value:
                backup = f"{self.config.state_file}.{saved_mode.lower()}_{int(time.time())}.bak"
                try:
                    os.replace(self.config.state_file, backup)
                except Exception as e:
                    logger.error(f"Could not back up mismatched state file: {e}")
                logger.warning(
                    f"STATE FILE MODE MISMATCH: file was written in {saved_mode} mode but running "
                    f"{self.config.mode.value}. Backed up to {backup}; starting from a CLEAN state."
                )
                self.save_state()
                return

            self.shadow_available_tmn = float(data.get("shadow_available_tmn") or data.get("available_tmn") or self.config.initial_capital_tmn)
            self.peak_equity_tmn = float(data.get("peak_equity_tmn") or self.config.initial_capital_tmn)
            self.baseline_equity_tmn = float(data.get("baseline_equity_tmn") or self.config.initial_capital_tmn)
            self.baseline_set = bool(data.get("baseline_set", self.config.mode != ExecutionMode.LIVE))
            self.realized_pnl_tmn = float(data.get("realized_pnl_tmn") or 0.0)
            self.total_trades_count = int(data.get("total_trades_count", 0))
            self.winning_trades_count = int(data.get("winning_trades_count", 0))
            self.losing_trades_count = int(data.get("losing_trades_count", 0))
            self.pending_buy_orders = data.get("pending_buy_orders") or {}
            self.last_exit_ts = {k: float(v) for k, v in (data.get("last_exit_ts") or {}).items()}
            self.consecutive_losses = int(data.get("consecutive_losses", 0))
            self.entries_paused_until_ts = float(data.get("entries_paused_until_ts", 0.0))

            pos_keys = OpenPosition.__annotations__.keys()
            for sym, pos_dict in data.get("positions", {}).items():
                filtered = {k: v for k, v in pos_dict.items() if k in pos_keys}
                self.positions[sym] = OpenPosition(**filtered)
            logger.info(f"Restored portfolio state: {len(self.positions)} active positions.")
        except Exception as e:
            logger.warning(f"Could not load state file: {e}")

    # ------------------------------------------------------------------ equity
    def compute_equity(self, client: WallexClient, latest_prices: Dict[str, float]) -> tuple[float, float]:
        if self.config.mode == ExecutionMode.SHADOW:
            pos_val = sum(
                pos.quantity * latest_prices.get(sym, pos.entry_price)
                for sym, pos in self.positions.items()
            )
            locked_pending_tmn = sum(
                p.get("order_tmn", 0.0) for p in self.pending_buy_orders.values()
            )
            total_eq = self.shadow_available_tmn + locked_pending_tmn + pos_val
            self.peak_equity_tmn = max(self.peak_equity_tmn, total_eq)
            return self.shadow_available_tmn, total_eq

        try:
            balances = client.get_balances()
            tmn_info = balances.get("TMN", {})
            tmn_value = float(tmn_info.get("value", 0.0) or 0.0)
            tmn_locked = float(tmn_info.get("locked", 0.0) or 0.0)
            avail_tmn = tmn_value - tmn_locked
            total_tmn = tmn_value

            crypto_val = 0.0
            changed = False
            for sym, rules in client.market_rules.items():
                base = rules["base_asset"]
                b_info = balances.get(base, {})
                qty = float(b_info.get("value", 0.0) or 0.0)
                free_qty = max(0.0, qty - float(b_info.get("locked", 0.0) or 0.0))
                min_qty = rules.get("min_qty", 0.00001)
                pos = self.positions.get(sym)
                px = latest_prices.get(sym) or (pos.entry_price if pos else 0.0)
                if qty > 0 and px > 0:
                    crypto_val += qty * px

                if pos:
                    if free_qty >= min_qty:
                        # Sync tracked quantity with actual post-fee wallet balance
                        pos.quantity = free_qty
                    elif qty < min_qty and _age_sec(pos.opened_at) > 120 and sym not in self.pending_buy_orders:
                        logger.error(
                            f"[PHANTOM POSITION] {sym} is tracked but the wallet holds {qty}. "
                            f"Order never filled or coins left the account. Removing from tracking."
                        )
                        self.positions.pop(sym, None)
                        changed = True
                elif px > 0 and qty * px >= self.config.exchange_min_notional_tmn and sym not in self.pending_buy_orders:
                    if sym not in self._orphan_warned:
                        self._orphan_warned.add(sym)
                        logger.warning(
                            f"[ORPHAN BALANCE] Wallet holds {qty} {base} (~{qty*px:,.0f} TMN) with no tracked position. "
                            f"It has no stop-loss; it will still be liquidated by the end-of-contest flatten."
                        )
            if changed:
                self.save_state()

            total_eq = total_tmn + crypto_val

            # First successful LIVE read on a fresh state: use the REAL wallet as baseline/peak
            if not self.baseline_set and not self.positions and self.total_trades_count == 0:
                self.baseline_equity_tmn = total_eq
                self.peak_equity_tmn = total_eq
                self.baseline_set = True
                self.save_state()
                logger.info(f"LIVE baseline equity set from wallet: {total_eq:,.0f} TMN")

            self.peak_equity_tmn = max(self.peak_equity_tmn, total_eq)
            return max(0.0, avail_tmn), total_eq
        except Exception as e:
            logger.error(f"Error computing live equity from Wallex: {e}")
            return 0.0, self.peak_equity_tmn

    # ------------------------------------------------------------------ exits
    def check_deterministic_exits(self, symbol: str, features: dict) -> Optional[AIDecisionOutput]:
        pos = self.positions.get(symbol)
        if not pos:
            return None

        current_bid = features["best_bid"]
        pnl_pct = (current_bid - pos.entry_price) / pos.entry_price

        if current_bid > pos.highest_price:
            pos.highest_price = current_bid

        if pnl_pct >= self.config.trailing_activation_pct:
            pos.trailing_active = True
            # Breakeven locks at least (Maker Buy + Taker Sell + 0.15% net profit margin) above entry price
            breakeven_plus = pos.entry_price * (1.0 + self.config.maker_fee_rate + self.config.taker_fee_rate + 0.0015)
            trail_stop = pos.highest_price * (1.0 - self.config.trailing_distance_pct)
            new_sl = max(pos.stop_loss_price, breakeven_plus, trail_stop)
            if new_sl > pos.stop_loss_price:
                pos.stop_loss_price = new_sl
                self.save_state()

        if current_bid <= pos.stop_loss_price:
            exit_type = "TRAILING_STOP_PROFIT_LOCK" if pos.trailing_active else "HARD_STOP_LOSS"
            return AIDecisionOutput(
                symbol=symbol,
                action="SELL",
                order_type="MARKET",
                confidence=1.0,
                allocation_pct=0.0,
                signals_observed=[exit_type, f"PNL_{pnl_pct*100:.2f}%"],
                reasoning_fa=(
                    f"خروج خودکار مدیریت ریسک ({exit_type}): قیمت فعلی ({current_bid:,.0f}) به سطح توقف "
                    f"({pos.stop_loss_price:,.0f}) رسید. بازدهی ناخالص پوزیشن: {pnl_pct*100:.2f}%."
                ),
                decision_source="RISK_ENGINE",
            )

        if current_bid >= pos.take_profit_price:
            return AIDecisionOutput(
                symbol=symbol,
                action="SELL",
                order_type="MARKET",
                confidence=1.0,
                allocation_pct=0.0,
                signals_observed=["TAKE_PROFIT_TARGET_HIT", f"PNL_{pnl_pct*100:.2f}%"],
                reasoning_fa=(
                    f"تحقق سود در هدف قیمتی (Take-Profit): قیمت ({current_bid:,.0f}) از حد سود تعیین‌شده "
                    f"({pos.take_profit_price:,.0f}) عبور کرد. بازدهی ناخالص پوزیشن: {pnl_pct*100:.2f}%."
                ),
                decision_source="RISK_ENGINE",
            )

        # Time stop: free dead capital instead of paying nothing-for-something over many hours
        if self.config.time_stop_sec > 0:
            held = _age_sec(pos.opened_at)
            if held >= self.config.time_stop_sec and pnl_pct < self.config.time_stop_min_pnl_pct:
                return AIDecisionOutput(
                    symbol=symbol,
                    action="SELL",
                    order_type="MARKET",
                    confidence=1.0,
                    allocation_pct=0.0,
                    signals_observed=["TIME_STOP", f"HELD_{held/3600:.1f}H", f"PNL_{pnl_pct*100:.2f}%"],
                    reasoning_fa=(
                        f"خروج زمانی: پوزیشن {held/3600:.1f} ساعت باز بوده و بازدهی ({pnl_pct*100:.2f}%) "
                        f"به حداقل هدف نرسیده است؛ سرمایه برای فرصت‌های بهتر آزاد شد."
                    ),
                    decision_source="RISK_ENGINE",
                )

        return None

    # ------------------------------------------------------------------ entries
    def entry_block_reason(self, sym: str) -> Optional[str]:
        now = time.time()
        end = self.config.contest_end_timestamp
        if end > 0 and (end - now) <= self.config.no_new_entries_before_end_sec:
            return "NO_NEW_ENTRIES_NEAR_CONTEST_END"
        if now < self.entries_paused_until_ts:
            return f"LOSS_STREAK_PAUSE_{int((self.entries_paused_until_ts - now) / 60)}M_LEFT"
        last = self.last_exit_ts.get(sym)
        if last and (now - last) < self.config.reentry_cooldown_sec:
            return f"REENTRY_COOLDOWN_{int((self.config.reentry_cooldown_sec - (now - last)) / 60)}M_LEFT"
        return None

    def register_exit(self, sym: str, net_pnl_tmn: float):
        now = time.time()
        self.last_exit_ts[sym] = now
        if net_pnl_tmn < 0:
            self.consecutive_losses += 1
            if self.consecutive_losses >= self.config.max_consecutive_losses:
                self.entries_paused_until_ts = now + self.config.loss_pause_sec
                logger.warning(
                    f"[LOSS STREAK] {self.consecutive_losses} consecutive losing trades -> new entries paused "
                    f"for {self.config.loss_pause_sec/60:.0f} min."
                )
                self.consecutive_losses = 0
        else:
            self.consecutive_losses = 0

    def approve_and_size_buy(
        self, decision: AIDecisionOutput, features: dict, avail_tmn: float, total_eq: float
    ) -> tuple[bool, str, float]:
        drawdown = (self.peak_equity_tmn - total_eq) / max(self.peak_equity_tmn, 1.0)
        if drawdown >= self.config.max_portfolio_drawdown_pct:
            return False, f"CIRCUIT_BREAKER: Drawdown {drawdown:.2%} >= {self.config.max_portfolio_drawdown_pct:.2%}", 0.0

        blocked = self.entry_block_reason(decision.symbol)
        if blocked:
            return False, blocked, 0.0

        total_open_count = len(self.positions) + len(self.pending_buy_orders)
        if total_open_count >= self.config.max_concurrent_positions:
            return False, "MAX_CONCURRENT_POSITIONS_REACHED", 0.0

        if decision.symbol in self.positions or decision.symbol in self.pending_buy_orders:
            return False, "ALREADY_IN_POSITION_OR_PENDING", 0.0

        if decision.confidence < 0.65:
            return False, f"LOW_CONFIDENCE_{decision.confidence}", 0.0

        if features["spread_pct"] > self.config.max_spread_pct:
            return False, f"WIDE_SPREAD_{features['spread_pct']}%", 0.0

        min_reserve = total_eq * self.config.min_cash_reserve_pct
        spendable = max(0.0, avail_tmn - min_reserve)
        alloc_pct = min(decision.allocation_pct or self.config.max_position_pct, self.config.max_position_pct)
        order_tmn = min(total_eq * alloc_pct, spendable)

        if order_tmn < self.config.min_order_tmn:
            return False, f"INSUFFICIENT_SPENDABLE_TMN ({order_tmn:,.0f} < {self.config.min_order_tmn:,.0f})", 0.0

        return True, "APPROVED_BY_RISK_GUARD", order_tmn

    def log_decision(
        self,
        symbol: str,
        features: dict,
        decision: AIDecisionOutput,
        risk_verdict: str,
        execution_result: Optional[dict],
        total_equity_tmn: float,
        trade_metrics: Optional[dict] = None,
    ):
        drawdown_pct = ((self.peak_equity_tmn - total_equity_tmn) / max(self.peak_equity_tmn, 1.0)) * 100.0
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "mode": self.config.mode.value,
            "symbol": symbol,
            "action": decision.action,
            "order_type": decision.order_type,
            "decision_source": decision.decision_source,
            "llm_latency_ms": decision.llm_latency_ms,
            "ai_confidence": decision.confidence,
            "signals_observed": decision.signals_observed,
            "reasoning_fa": decision.reasoning_fa,
            "risk_verdict": risk_verdict,
            "portfolio_equity_tmn": round(total_equity_tmn, 0),
            "current_drawdown_pct": round(drawdown_pct, 3),
            "realized_pnl_tmn_cumulative": round(self.realized_pnl_tmn, 0),
            "trade_metrics": trade_metrics,
            "market_snapshot": features,
            "execution_result": execution_result,
        }
        try:
            with open(self.config.decision_log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.error(f"Failed to append to decision log: {e}")


class TradingAgentRunner:
    def __init__(self, config: AgentConfig):
        self.config = config
        self.client = WallexClient(config)
        self.risk_mgr = PortfolioAndRiskManager(config)
        self.strategist = CostEffectiveLLMStrategist(config)
        self.running = True
        self.pending_order_timestamps: Dict[str, float] = {}
        self.last_total_eq: float = config.initial_capital_tmn
        self.flatten_attempts: int = 0

        signal.signal(signal.SIGINT, self._handle_stop)
        signal.signal(signal.SIGTERM, self._handle_stop)

    def _handle_stop(self, signum, frame):
        logger.info("Shutdown signal received. Saving portfolio state cleanly...")
        self.risk_mgr.save_state()
        self.running = False

    # ------------------------------------------------------------------ pending orders
    def sweep_stale_open_orders(self):
        now = time.time()

        if self.config.mode == ExecutionMode.LIVE:
            if not self.risk_mgr.pending_buy_orders and not self.pending_order_timestamps:
                return
            open_orders = self.client.get_open_orders()
            if open_orders is None:
                # API call failed temporarily; do NOT assume orders filled!
                return

            live_cids = set()
            for o in open_orders:
                cid = _order_cid(o)
                if not cid:
                    continue
                live_cids.add(cid)
                first_seen = self.pending_order_timestamps.setdefault(cid, now)
                if now - first_seen >= self.config.stale_order_timeout_sec:
                    logger.warning(f"[STALE ORDER] Canceling stale order {cid} on {o.get('symbol')}")
                    self.client.cancel_order(cid)
                    self.pending_order_timestamps.pop(cid, None)
                    for sym in list(self.risk_mgr.pending_buy_orders.keys()):
                        if self.risk_mgr.pending_buy_orders[sym].get("client_id") == cid:
                            del self.risk_mgr.pending_buy_orders[sym]
                            self.risk_mgr.save_state()

            # Verify pending orders that left open_orders using actual wallet balance
            for sym, pending in list(self.risk_mgr.pending_buy_orders.items()):
                cid = pending.get("client_id")
                if cid and cid not in live_cids:
                    free_qty = self.client.get_free_asset_balance(sym)
                    min_qty = self.client.market_rules.get(sym, {}).get("min_qty", 0.00001)
                    if free_qty >= min_qty:
                        logger.info(f"[LIMIT FILLED] Verified wallet balance {free_qty} for {sym}. Promoting to position.")
                        self.risk_mgr.positions[sym] = OpenPosition(
                            symbol=sym,
                            entry_price=pending["entry_price"],
                            quantity=free_qty,
                            stop_loss_price=pending["stop_loss_price"],
                            take_profit_price=pending["take_profit_price"],
                            highest_price=pending["entry_price"],
                            opened_at=datetime.now(timezone.utc).isoformat(),
                            client_id=cid,
                            cost_tmn=pending.get("order_tmn", 0.0),
                            entry_fee_rate=self.config.maker_fee_rate,
                        )
                    else:
                        status = self.client.get_order_by_client_id(cid)
                        logger.warning(
                            f"[ORDER EXPIRED/CANCELED] Order {cid} on {sym} no longer open and wallet balance is 0. "
                            f"Exchange status: {status}"
                        )
                    self.pending_order_timestamps.pop(cid, None)
                    del self.risk_mgr.pending_buy_orders[sym]
                    self.risk_mgr.save_state()

        else:
            for sym, pending in list(self.risk_mgr.pending_buy_orders.items()):
                created_ts = pending.get("created_at_ts", now)
                if now - created_ts >= self.config.stale_order_timeout_sec:
                    logger.info(f"[SHADOW STALE] Released locked shadow TMN for {sym}")
                    self.risk_mgr.shadow_available_tmn += pending.get("order_tmn", 0.0)
                    del self.risk_mgr.pending_buy_orders[sym]
                    self.risk_mgr.save_state()

    # ------------------------------------------------------------------ end of contest
    def execute_flatten_all_to_tmn(self) -> bool:
        """Liquidates everything. Returns True when no tracked positions remain and no sell errors occurred."""
        logger.warning("=" * 65)
        logger.warning("=== EXECUTING END-OF-CONTEST COMPLETE FLATTEN TO REALIZED TMN ===")
        logger.warning("=" * 65)
        all_ok = True

        if self.config.mode == ExecutionMode.LIVE:
            open_orders = self.client.get_open_orders() or []
            for o in open_orders:
                cid = _order_cid(o)
                if cid:
                    self.client.cancel_order(cid)
            self.risk_mgr.pending_buy_orders.clear()
        else:
            for sym, p in list(self.risk_mgr.pending_buy_orders.items()):
                self.risk_mgr.shadow_available_tmn += p.get("order_tmn", 0.0)
                del self.risk_mgr.pending_buy_orders[sym]
            self.risk_mgr.save_state()

        for sym, pos in list(self.risk_mgr.positions.items()):
            try:
                ob = self.client.get_orderbook(sym)
                best_bid = float(ob["bid"][0]["price"]) if ob.get("bid") else pos.entry_price
                decision = AIDecisionOutput(
                    symbol=sym,
                    action="SELL",
                    order_type="MARKET",
                    confidence=1.0,
                    allocation_pct=0.0,
                    signals_observed=["CONTEST_DEADLINE_REALIZE_TMN"],
                    reasoning_fa="نقد کردن خودکار دارایی در پایان چالش جهت تحقق کامل سود به تومان طبق دستورالعمل داوری والکس.",
                    decision_source="RISK_ENGINE",
                )
                self._execute_sell(
                    sym, pos, {"best_bid": best_bid}, decision, "END_OF_CONTEST_LIQUIDATION", self.last_total_eq
                )
            except Exception as e:
                all_ok = False
                logger.error(f"Failed to liquidate {sym}: {e}")

        if self.config.mode == ExecutionMode.LIVE:
            try:
                balances = self.client.get_balances()
                for asset, info in balances.items():
                    if asset == "TMN":
                        continue
                    free_val = float(info.get("value", 0.0) or 0.0) - float(info.get("locked", 0.0) or 0.0)
                    sym = f"{asset}TMN"
                    if sym in self.client.market_rules and free_val > 0:
                        ob = self.client.get_orderbook(sym)
                        best_bid = float(ob["bid"][0]["price"]) if ob.get("bid") else 0.0
                        notional_tmn = float(self.client.format_quantity(sym, free_val)) * best_bid
                        # Only liquidate if notional value is >= 100,000 TMN
                        if notional_tmn >= self.config.exchange_min_notional_tmn:
                            logger.info(
                                f"Liquidating remaining wallet asset {asset} "
                                f"({free_val} worth {notional_tmn:,.0f} TMN)..."
                            )
                            try:
                                self.client.place_spot_order(
                                    sym, "SELL", "MARKET", free_val, reference_price=best_bid
                                )
                            except Exception as e:
                                all_ok = False
                                logger.error(f"Wallet liquidation of {asset} failed: {e}")
                        else:
                            logger.info(
                                f"Skipping dust balance on {asset} ({notional_tmn:,.0f} TMN < 100,000 TMN min)."
                            )
            except Exception as e:
                all_ok = False
                logger.error(f"Error checking remaining live wallet balances: {e}")

        remaining = len(self.risk_mgr.positions)
        if remaining:
            all_ok = False
        logger.info(
            f"Flatten pass finished | positions left: {remaining} | ok={all_ok}"
        )
        return all_ok

    def _deadline_reached(self) -> bool:
        """Starts/continues the end-of-contest flatten. Returns True if the cycle must stop here."""
        end = self.config.contest_end_timestamp
        if end <= 0:
            return False
        remaining = end - int(time.time())
        if remaining > self.config.flatten_before_end_sec:
            return False

        self.flatten_attempts += 1
        ok = self.execute_flatten_all_to_tmn()
        if ok or self.flatten_attempts >= 6 or remaining <= 0:
            if not ok:
                logger.error("Flatten finished with problems; check wallet manually!")
            self._log_final_summary()
            self.running = False
        return True

    def _log_final_summary(self):
        rm = self.risk_mgr
        logger.info(
            f"FINAL SUMMARY | Realized PnL: {rm.realized_pnl_tmn:+,.0f} TMN | Trades: {rm.total_trades_count} "
            f"(W {rm.winning_trades_count} / L {rm.losing_trades_count}) | Peak equity: {rm.peak_equity_tmn:,.0f}"
        )
        rm.save_state()

    # ------------------------------------------------------------------ cycle helpers
    def check_positions_fast(self):
        """Stop-loss / trailing / take-profit / time-stop on order-book data only (cheap, runs every few seconds)."""
        if not self.risk_mgr.positions:
            return
        for sym, pos in list(self.risk_mgr.positions.items()):
            try:
                ob = self.client.get_orderbook(sym)
                bids, asks = ob.get("bid", []), ob.get("ask", [])
                if not bids or not asks:
                    continue
                best_bid = float(bids[0]["price"])
                best_ask = float(asks[0]["price"])
            except Exception as e:
                logger.warning(f"Fast position check failed for {sym}: {e}")
                continue
            feats = {"best_bid": best_bid, "best_ask": best_ask, "last_price": best_bid, "light_snapshot": True}
            det_exit = self.risk_mgr.check_deterministic_exits(sym, feats)
            if det_exit is not None:
                self._execute_sell(sym, pos, feats, det_exit, "DETERMINISTIC_RISK_EXIT", self.last_total_eq)

    def _fetch_symbol_features(self, sym: str) -> Tuple[str, Optional[dict]]:
        try:
            ob = self.client.get_orderbook(sym)
            df = self.client.get_candles_cached(sym, resolution=self.config.candle_resolution, lookback_hours=72)
            trades = self.client.get_latest_trades(sym)
            feats = QuantFeatureEngine.compute_features(df, ob, trades)
            if not feats.get("valid"):
                return sym, None
            return sym, feats
        except Exception as e:
            logger.warning(f"Market data fetch warning on {sym}: {e}")
            return sym, None

    def _market_regime_ok(self, snapshots: Dict[str, dict]) -> Tuple[bool, str]:
        """Alts follow BTC. Don't open new longs while BTC itself is dumping / in a short-term downtrend."""
        if not self.config.btc_regime_filter:
            return True, "REGIME_FILTER_OFF"
        btc = snapshots.get("BTCTMN")
        if not btc:
            return True, "BTC_DATA_UNAVAILABLE"
        if btc.get("roc_3_pct", 0.0) <= -0.5:
            return False, f"BTC_DUMP_ROC3_{btc.get('roc_3_pct')}%"
        if btc.get("last_price", 0.0) < btc.get("ema21", 0.0) and btc.get("ema9", 0.0) < btc.get("ema21", 0.0):
            return False, "BTC_SHORT_TERM_DOWNTREND"
        return True, "BTC_OK"

    @staticmethod
    def _best_fill_price(order: Optional[dict], fallback: float) -> float:
        if not order:
            return fallback
        for k in ("executedPrice", "avgPrice", "averagePrice", "price"):
            try:
                p = float(order.get(k))
            except (TypeError, ValueError):
                continue
            if p > 0 and abs(p - fallback) / fallback < 0.01:
                return p
        return fallback

    # ------------------------------------------------------------------ main cycle
    def run_single_cycle(self, active_markets: List[str]):
        if self._deadline_reached():
            return

        self.sweep_stale_open_orders()

        # PASS 1: protect open positions first (order book only) — never wait behind candles / LLM
        self.check_positions_fast()

        # PASS 2: market data in parallel, then signals
        latest_prices: Dict[str, float] = {}
        market_snapshots: Dict[str, dict] = {}
        with ThreadPoolExecutor(max_workers=max(1, self.config.fetch_workers)) as pool:
            for sym, feats in pool.map(self._fetch_symbol_features, active_markets):
                if feats is not None:
                    latest_prices[sym] = feats["best_bid"]
                    market_snapshots[sym] = feats

        avail_tmn, total_eq = self.risk_mgr.compute_equity(self.client, latest_prices)
        self.last_total_eq = total_eq
        baseline = self.risk_mgr.baseline_equity_tmn or self.config.initial_capital_tmn
        net_return_pct = ((total_eq - baseline) / baseline) * 100.0
        regime_ok, regime_reason = self._market_regime_ok(market_snapshots)
        logger.info(
            f"Cycle Portfolio | Mode: {self.config.mode.value} | Equity: {total_eq:,.0f} TMN "
            f"({net_return_pct:+.2f}%) | Cash: {avail_tmn:,.0f} TMN | Open Pos: {len(self.risk_mgr.positions)} "
            f"| Pending Orders: {len(self.risk_mgr.pending_buy_orders)} | Regime: {regime_reason} "
            f"| Data ok: {len(market_snapshots)}/{len(active_markets)}"
        )

        for sym, feats in market_snapshots.items():
            if not self.running:
                break

            # 1. In SHADOW mode, check if resting limit order was touched/crossed
            if self.config.mode == ExecutionMode.SHADOW and sym in self.risk_mgr.pending_buy_orders:
                pending = self.risk_mgr.pending_buy_orders[sym]
                # Conservative: fill only when the ask actually reaches our limit (no candle-close optimism)
                if feats["best_ask"] <= pending["limit_price"]:
                    fill_price = min(pending["limit_price"], feats["best_ask"])
                    logger.info(f"[SHADOW LIMIT FILLED] {sym} filled @ {fill_price:,.0f} TMN")
                    self.risk_mgr.positions[sym] = OpenPosition(
                        symbol=sym,
                        entry_price=fill_price,
                        quantity=pending["quantity"],
                        stop_loss_price=pending["stop_loss_price"],
                        take_profit_price=pending["take_profit_price"],
                        highest_price=fill_price,
                        opened_at=datetime.now(timezone.utc).isoformat(),
                        client_id=pending["client_id"],
                        cost_tmn=pending.get("order_tmn", 0.0),
                        entry_fee_rate=self.config.maker_fee_rate,
                    )
                    del self.risk_mgr.pending_buy_orders[sym]
                    self.risk_mgr.save_state()

            pos = self.risk_mgr.positions.get(sym)

            # 2. Deterministic exits again on the full snapshot (SL / Trailing / TP / Time-stop)
            if pos is not None:
                det_exit = self.risk_mgr.check_deterministic_exits(sym, feats)
                if det_exit is not None:
                    self._execute_sell(sym, pos, feats, det_exit, "DETERMINISTIC_RISK_EXIT", total_eq)
                    continue

            # 3. Capacity / regime / cooldown filters: don't waste LLM calls on entries we'd reject anyway
            total_open_count = len(self.risk_mgr.positions) + len(self.risk_mgr.pending_buy_orders)
            if pos is None:
                if total_open_count >= self.config.max_concurrent_positions:
                    continue
                min_reserve = total_eq * self.config.min_cash_reserve_pct
                spendable = max(0.0, avail_tmn - min_reserve)
                if spendable < self.config.min_order_tmn:
                    continue
                if not regime_ok:
                    continue
                if self.risk_mgr.entry_block_reason(sym):
                    continue

            # 4. Consult AI Strategist (if Quant Gatekeeper triggers)
            unrealized_pct = ((feats["best_bid"] - pos.entry_price) / pos.entry_price * 100.0) if pos else 0.0
            time_held_sec = _age_sec(pos.opened_at) if pos else 0.0

            portfolio_context = {
                "available_tmn": avail_tmn,
                "total_equity_tmn": total_eq,
                "position_qty": pos.quantity if pos else 0.0,
                "entry_price": pos.entry_price if pos else None,
                "unrealized_pnl_pct": round(unrealized_pct, 2),
                "time_held_sec": time_held_sec,
                "max_positions_reached": total_open_count >= self.config.max_concurrent_positions,
            }
            recent_flow = {"taker_buy_ratio": feats["taker_buy_ratio"], "obi_top5": feats["obi_top5"]}

            decision, llm_invoked = self.strategist.evaluate_market(sym, feats, portfolio_context, recent_flow)

            if decision.action == "BUY" and pos is None and sym not in self.risk_mgr.pending_buy_orders:
                approved, reason, order_tmn = self.risk_mgr.approve_and_size_buy(
                    decision, feats, avail_tmn, total_eq
                )
                if approved:
                    self._execute_buy(sym, feats, decision, order_tmn, reason, total_eq)
                    avail_tmn -= order_tmn
                else:
                    self.risk_mgr.log_decision(sym, feats, decision, f"REJECTED: {reason}", None, total_eq)

            elif decision.action == "SELL" and pos is not None:
                # Never allow premature tactical exit within 20 minutes if in minor loss (above SL).
                # Fee drag (0.55%) is the #1 enemy. Let deterministic SL/TP manage the trade!
                if time_held_sec < 1200 and unrealized_pct < 1.0:
                    logger.info(
                        f"[TACTICAL SELL BLOCKED] {sym} held for only {time_held_sec/60:.1f}m with PnL "
                        f"{unrealized_pct:+.2f}%. Protecting trade from fee drag; leaving to SL/TP."
                    )
                else:
                    self._execute_sell(sym, pos, feats, decision, "AI_TACTICAL_EXIT", total_eq)

            elif decision.action == "HOLD" and llm_invoked:
                # Log genuine AI HOLD evaluations so judges see active AI filtering
                self.risk_mgr.log_decision(sym, feats, decision, "AI_EVALUATED_HOLD", None, total_eq)

    # ------------------------------------------------------------------ execution
    def _execute_buy(
        self, sym: str, feats: dict, decision: AIDecisionOutput, order_tmn: float, risk_reason: str, total_eq: float
    ):
        use_market = self.config.force_market_entries or decision.order_type == "MARKET"
        order_type = "MARKET" if use_market else "LIMIT"

        if use_market:
            exec_price = feats["best_ask"]
        else:
            # Place limit buy at best_bid (or AI limit price capped at best_ask)
            exec_price = min(decision.limit_price or feats["best_bid"], feats["best_ask"])

        # Wallex Credit Account: Maker (0.25%) for LIMIT buy, Taker (0.30%) for MARKET buy
        buy_fee_rate = self.config.taker_fee_rate if use_market else self.config.maker_fee_rate
        raw_qty = (order_tmn * (1.0 - buy_fee_rate)) / exec_price

        sl_price, tp_price = compute_sl_tp(
            self.config, exec_price, feats["atr_14"], decision.stop_loss_price, decision.take_profit_price
        )

        try:
            formatted_qty_float = float(self.client.format_quantity(sym, raw_qty))
            post_round_notional = formatted_qty_float * exec_price
            if post_round_notional < self.config.exchange_min_notional_tmn:
                logger.warning(
                    f"[SKIPPED BUY] {sym} post-rounding notional ({post_round_notional:,.0f} TMN) "
                    f"< {self.config.exchange_min_notional_tmn:,.0f} TMN minimum."
                )
                return

            res = self.client.place_spot_order(
                symbol=sym,
                side="BUY",
                order_type=order_type,
                quantity=raw_qty,
                price=exec_price if order_type == "LIMIT" else None,
                reference_price=exec_price,
            )
            filled_qty = formatted_qty_float
            client_id = res.get("clientOrderId") or res.get("_sent_client_id") or f"wlx_{int(time.time()*1000)}"

            if self.config.mode == ExecutionMode.SHADOW:
                self.risk_mgr.shadow_available_tmn -= order_tmn

            # Market orders or marketable limit orders fill immediately; passive limits wait in pending_buy_orders
            is_immediate_fill = (order_type == "MARKET" or exec_price >= feats["best_ask"])

            if is_immediate_fill:
                actual_qty = filled_qty
                entry_price = exec_price
                if self.config.mode == ExecutionMode.LIVE:
                    min_qty = self.client.market_rules.get(sym, {}).get("min_qty", 0.00001)
                    wallet_qty = 0.0
                    for _ in range(5):
                        time.sleep(0.6)
                        wallet_qty = self.client.get_free_asset_balance(sym)
                        if wallet_qty >= min_qty:
                            break
                    if wallet_qty >= min_qty:
                        actual_qty = wallet_qty
                    else:
                        logger.error(
                            f"[UNCONFIRMED FILL] {sym}: wallet shows {wallet_qty} after market buy. "
                            f"Tracking estimated qty; reconciliation will drop it if it never appears."
                        )
                    order_info = self.client.get_order_by_client_id(client_id)
                    entry_price = self._best_fill_price(order_info or res, exec_price)

                self.risk_mgr.positions[sym] = OpenPosition(
                    symbol=sym,
                    entry_price=entry_price,
                    quantity=actual_qty,
                    stop_loss_price=round(sl_price, 6),
                    take_profit_price=round(tp_price, 6),
                    highest_price=entry_price,
                    opened_at=datetime.now(timezone.utc).isoformat(),
                    client_id=client_id,
                    cost_tmn=order_tmn,
                    entry_fee_rate=buy_fee_rate,
                )
            else:
                self.risk_mgr.pending_buy_orders[sym] = {
                    "client_id": client_id,
                    "entry_price": exec_price,
                    "limit_price": exec_price,
                    "quantity": filled_qty,
                    "stop_loss_price": round(sl_price, 6),
                    "take_profit_price": round(tp_price, 6),
                    "order_tmn": order_tmn,
                    "created_at_ts": time.time(),
                }
                self.pending_order_timestamps[client_id] = time.time()

            self.risk_mgr.save_state()
            trade_meta = {
                "allocated_tmn": round(order_tmn, 0),
                "entry_price": exec_price,
                "stop_loss_price": round(sl_price, 6),
                "take_profit_price": round(tp_price, 6),
                "stop_distance_pct": round((exec_price - sl_price) / exec_price * 100.0, 3),
                "take_profit_distance_pct": round((tp_price - exec_price) / exec_price * 100.0, 3),
                "executed_order_type": order_type,
            }
            self.risk_mgr.log_decision(sym, feats, decision, risk_reason, res, total_eq, trade_meta)
            logger.info(
                f"[BUY SUBMITTED] {sym} | Type: {order_type} | Qty: {filled_qty} @ {exec_price:,.0f} TMN | "
                f"SL: {sl_price:,.0f} | TP: {tp_price:,.0f} | Source: {decision.decision_source}"
            )
        except Exception as e:
            logger.error(f"Order placement failed on {sym}: {e}")

    def _execute_sell(
        self, sym: str, pos: OpenPosition, feats: dict, decision: AIDecisionOutput, exit_reason: str, total_eq: float
    ):
        exec_price = feats["best_bid"]
        try:
            sell_qty = pos.quantity
            # In LIVE mode, reconcile with actual free wallet balance so fee deduction never causes 400 error
            if self.config.mode == ExecutionMode.LIVE:
                wallet_free = self.client.get_free_asset_balance(sym)
                min_qty = self.client.market_rules.get(sym, {}).get("min_qty", 0.00001)
                if wallet_free < min_qty:
                    logger.warning(f"Wallet free balance for {sym} ({wallet_free}) < min_qty. Removing stale position.")
                    self.risk_mgr.positions.pop(sym, None)
                    self.risk_mgr.save_state()
                    return
                sell_qty = min(pos.quantity, wallet_free)

            sell_notional_tmn = float(self.client.format_quantity(sym, sell_qty)) * exec_price
            if sell_notional_tmn < self.config.exchange_min_notional_tmn:
                logger.warning(
                    f"[DUST POSITION IGNORED] {sym} value ({sell_notional_tmn:,.0f} TMN) is below "
                    f"Wallex 100,000 TMN minimum. Removing from active tracking."
                )
                self.risk_mgr.positions.pop(sym, None)
                self.risk_mgr.save_state()
                return

            # ALWAYS MARKET: a resting limit sell would leave coins held but untracked (no stop-loss).
            order_type = "MARKET"
            res = self.client.place_spot_order(
                symbol=sym,
                side="SELL",
                order_type=order_type,
                quantity=sell_qty,
                price=None,
                reference_price=exec_price,
            )

            frac = min(1.0, sell_qty / pos.quantity) if pos.quantity > 0 else 1.0
            cost_full = pos.cost_tmn if pos.cost_tmn > 0 else pos.quantity * pos.entry_price * (1.0 + pos.entry_fee_rate)
            cost_part = cost_full * frac
            gross_proceeds = sell_qty * exec_price
            exit_fee = gross_proceeds * self.config.taker_fee_rate
            net_pnl_tmn = (gross_proceeds - exit_fee) - cost_part
            net_pnl_pct = (net_pnl_tmn / max(cost_part, 1.0)) * 100.0

            if self.config.mode == ExecutionMode.SHADOW:
                self.risk_mgr.shadow_available_tmn += (gross_proceeds - exit_fee)

            self.risk_mgr.realized_pnl_tmn += net_pnl_tmn
            self.risk_mgr.total_trades_count += 1
            if net_pnl_tmn > 0:
                self.risk_mgr.winning_trades_count += 1
            else:
                self.risk_mgr.losing_trades_count += 1

            self.risk_mgr.positions.pop(sym, None)
            if exit_reason != "END_OF_CONTEST_LIQUIDATION":
                self.risk_mgr.register_exit(sym, net_pnl_tmn)
            self.risk_mgr.save_state()

            trade_meta = {
                "entry_price": pos.entry_price,
                "exit_price": exec_price,
                "sold_quantity": sell_qty,
                "realized_net_pnl_tmn": round(net_pnl_tmn, 0),
                "realized_net_pnl_pct": round(net_pnl_pct, 3),
                "cumulative_trades": self.risk_mgr.total_trades_count,
                "win_rate_pct": round(
                    (self.risk_mgr.winning_trades_count / max(1, self.risk_mgr.total_trades_count)) * 100.0, 1
                ),
            }
            self.risk_mgr.log_decision(sym, feats, decision, exit_reason, res, total_eq, trade_meta)
            logger.info(
                f"[SELL EXECUTED] {sym} @ {exec_price:,.0f} TMN | Net PnL: {net_pnl_tmn:+,.0f} TMN "
                f"({net_pnl_pct:+.2f}%) | Reason: {exit_reason}"
            )
        except Exception as e:
            logger.error(f"Sell execution failed on {sym}: {e}")

    # ------------------------------------------------------------------ run loop
    def start(self):
        active_markets = self.client.load_market_rules()
        logger.info(f"Starting Wallex AI Agent in {self.config.mode.value} mode...")
        logger.info(
            f"Tracking {len(active_markets)} compliant markets | "
            f"Resolution: {self.config.candle_resolution}m | Capital: {self.config.initial_capital_tmn:,.0f} TMN | "
            f"State: {self.config.state_file} | Log: {self.config.decision_log_file}"
        )
        if self.config.contest_end_timestamp > 0:
            end = datetime.fromtimestamp(self.config.contest_end_timestamp, timezone.utc)
            left_h = (self.config.contest_end_timestamp - time.time()) / 3600.0
            logger.info(
                f"Contest end: {end.isoformat()} ({left_h:.1f}h left) | flatten {self.config.flatten_before_end_sec/60:.0f} min before | "
                f"no new entries in the last {self.config.no_new_entries_before_end_sec/60:.0f} min"
            )
        else:
            logger.warning("CONTEST_END_TIMESTAMP is not set: end-of-contest flatten is DISABLED!")

        last_full_cycle = 0.0
        while self.running:
            tick_start = time.time()
            try:
                if (tick_start - last_full_cycle) >= self.config.loop_interval_sec or last_full_cycle == 0.0:
                    self.run_single_cycle(active_markets)
                    last_full_cycle = time.time()
                else:
                    if self._deadline_reached():
                        pass
                    else:
                        self.check_positions_fast()
            except Exception as e:
                logger.error(f"Unexpected cycle error (auto-recovering): {e}", exc_info=True)
                last_full_cycle = time.time()

            if self.running:
                time.sleep(max(2.0, self.config.position_check_interval_sec - (time.time() - tick_start)))


def run_smoke_test(client: WallexClient, symbol: str):
    """
    One minimum-size LIVE round trip to verify API assumptions before the real run:
    field names, precision, fee side, fill speed. Costs roughly 0.6% of ~120k TMN plus spread.
    """
    cfg = client.config
    if cfg.mode != ExecutionMode.LIVE:
        print("Smoke test only runs in LIVE mode (use --mode LIVE).")
        return
    sym = MarketComplianceGuard.assert_market_allowed(symbol)
    if sym not in client.market_rules:
        print(f"{sym} is not in the compliant market list: {list(client.market_rules)}")
        return

    notional = cfg.exchange_min_notional_tmn * 1.2
    ob = client.get_orderbook(sym)
    ask = float(ob["ask"][0]["price"])
    qty = notional / ask
    base = client.market_rules[sym]["base_asset"]

    bal0 = client.get_balances()
    tmn0 = float(bal0.get("TMN", {}).get("value", 0) or 0)
    coin0 = float(bal0.get(base, {}).get("value", 0) or 0)
    print(f"[1] BEFORE  TMN={tmn0:,.0f}  {base}={coin0}")

    res = client.place_spot_order(sym, "BUY", "MARKET", qty, reference_price=ask)
    print(f"[2] BUY raw response: {json.dumps(res, ensure_ascii=False, default=str)}")
    time.sleep(2.0)

    bal1 = client.get_balances()
    tmn1 = float(bal1.get("TMN", {}).get("value", 0) or 0)
    coin1 = float(bal1.get(base, {}).get("value", 0) or 0)
    got = coin1 - coin0
    print(f"[3] AFTER BUY  TMN={tmn1:,.0f} (spent {tmn0 - tmn1:,.0f})  {base}={coin1} (received {got})")
    if qty > 0 and got > 0:
        print(f"    requested qty={client.format_quantity(sym, qty)} -> received {got} "
              f"=> implied fee taken from coin: {(1 - got / float(client.format_quantity(sym, qty)))*100:.3f}%")
    print(f"    spent TMN vs notional {qty*ask:,.0f}: implied TMN-side cost {(tmn0 - tmn1) / (qty*ask) - 1:+.4%}")

    cid = res.get("clientOrderId")
    print(f"[4] order lookup by client id: {client.get_order_by_client_id(cid)}")
    print(f"    open orders: {client.get_open_orders()}")

    if got <= 0:
        print("No coins received: check the response above. NOT selling.")
        return
    ob2 = client.get_orderbook(sym)
    bid = float(ob2["bid"][0]["price"])
    sell_res = client.place_spot_order(sym, "SELL", "MARKET", got, reference_price=bid)
    print(f"[5] SELL raw response: {json.dumps(sell_res, ensure_ascii=False, default=str)}")
    time.sleep(2.0)
    bal2 = client.get_balances()
    tmn2 = float(bal2.get("TMN", {}).get("value", 0) or 0)
    coin2 = float(bal2.get(base, {}).get("value", 0) or 0)
    print(f"[6] AFTER SELL TMN={tmn2:,.0f}  {base}={coin2}")
    print(f"    ROUND-TRIP COST: {tmn0 - tmn2:,.0f} TMN ({(tmn0 - tmn2) / (qty*ask) * 100:.3f}% of notional); "
          f"expected about 0.60% fees + spread")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Wallex Hackathon AI Spot Trading Agent")
    parser.add_argument("--mode", choices=["SHADOW", "LIVE"], default=None, help="Override execution mode")
    parser.add_argument("--flatten", action="store_true", help="Immediately liquidate all crypto holdings to TMN")
    parser.add_argument(
        "--smoke-test", nargs="?", const="DOGETMN", default=None, metavar="SYMBOL",
        help="LIVE only: one minimum-size BUY+SELL round trip to verify API assumptions, then exit",
    )
    args = parser.parse_args()

    cfg = AgentConfig()
    if args.mode:
        cfg.mode = ExecutionMode(args.mode)

    if args.smoke_test:
        smoke_client = WallexClient(cfg)
        smoke_client.load_market_rules()
        run_smoke_test(smoke_client, args.smoke_test)
        sys.exit(0)

    runner = TradingAgentRunner(cfg)

    if args.flatten:
        runner.client.load_market_rules()
        runner.execute_flatten_all_to_tmn()
    else:
        runner.start()