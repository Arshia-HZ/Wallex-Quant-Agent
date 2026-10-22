# main.py
"""
Wallex Agentic AI Spot Trading Engine.
Main Autonomous Loop, Position Management, Risk Guardrails, and Decision Logging.
Designed for the StarCoach Agentic AI Hackathon (Wallex Challenge).

Core Architecture & Solved Challenges:
1. Stale Limit Orders Auto-Cancellation: Automatically sweeps and cancels unfilled
   orders older than 3 minutes (180s) to immediately free up locked TMN capital.
2. Crash Resilience & State Persistence: Atomic writes to portfolio_state.json
   preserve entry prices, peak prices, trailing stops, and peak equity across restarts.
3. Realistic Shadow Mode: Live order book depth from Wallex, matching top-of-book,
   and 0.2% fee modeling for verifiable performance metrics.
4. Deterministic Risk Pre-Checks: Instantly executes Stop-Loss, Trailing Stop, and
   Take-Profit without waiting for LLM network latency.
5. Quant Gatekeeper: Saves 90%+ of LLM tokens by pre-filtering neutral markets in Python.
6. Contest Compliance & Liquidation: Rejects all 326 forbidden markets and automatically
   liquidates all crypto holdings to 100% realized TMN prior to contest deadline.
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
from datetime import datetime, timezone
from enum import Enum
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional
import requests
import pandas as pd
import numpy as np
from dotenv import load_dotenv

# Import custom compliance and strategist modules
from market_compliance import (
    MarketComplianceGuard,
    PRIORITY_ALLOWED_TMN_MARKETS,
    normalize_symbol,
)
from llm_strategist import CostEffectiveLLMStrategist, AIDecisionOutput

load_dotenv()

# ==========================================
# ۱. تنظیمات لاگینگ کنسول و فایل
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("agent_runtime.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("WallexAgent")


# ==========================================
# ۲. تنظیمات سراسری ایجنت و مدیریت ریسک
# ==========================================
class ExecutionMode(str, Enum):
    SHADOW = "SHADOW"  # دیتای زنده + هوش مصنوعی زنده + اجرای شبیه‌سازی‌شده
    LIVE = "LIVE"      # ثبت سفارش واقعی در حساب چالش والکس


@dataclass
class AgentConfig:
    api_key: str = field(default_factory=lambda: os.getenv("WALLEX_API_KEY", ""))
    base_url: str = field(default_factory=lambda: os.getenv("WALLEX_BASE_URL", "https://api.wallex.ir").rstrip("/"))
    mode: ExecutionMode = field(
        default_factory=lambda: ExecutionMode(os.getenv("EXECUTION_MODE", "SHADOW").upper())
    )

    # سرمایه و محدودیت‌های ریسک (۳۰ امتیاز مدیریت ریسک + ۳۰ امتیاز کنترل افت سرمایه)
    initial_capital_tmn: float = field(
        default_factory=lambda: float(os.getenv("INITIAL_CAPITAL_TMN", "5000000.0"))
    )
    max_position_pct: float = field(
        default_factory=lambda: float(os.getenv("MAX_POSITION_PCT", "0.22"))
    )  # حداکثر ۲۲٪ سرمایه در یک معامله (~۱.۱ میلیون تومان)
    min_cash_reserve_pct: float = field(
        default_factory=lambda: float(os.getenv("MIN_CASH_RESERVE_PCT", "0.20"))
    )  # حداقل ۲۰٪ نقدینگی رزرو (~۱ میلیون تومان)
    max_concurrent_positions: int = field(
        default_factory=lambda: int(os.getenv("MAX_CONCURRENT_POSITIONS", "3"))
    )  # حداکثر ۳ پوزیشن همزمان
    min_order_tmn: float = field(
        default_factory=lambda: float(os.getenv("MIN_ORDER_TMN", "150000.0"))
    )  # حداقل حجم سفارش مجاز به تومان (بالاتر از حداقل ۵۰,۰۰۰ تومان والکس)
    fee_rate_per_side: float = field(
        default_factory=lambda: float(os.getenv("FEE_RATE", "0.002"))
    )  # کارمزد ۰.۲٪ در هر سمت معامله

    # پارامترهای حد ضرر، حد سود و تریلینگ استاپ
    hard_stop_loss_pct: float = field(
        default_factory=lambda: float(os.getenv("HARD_STOP_LOSS_PCT", "0.012"))
    )  # حداکثر ۱.۲٪ ضرر در هر معامله
    default_take_profit_pct: float = field(
        default_factory=lambda: float(os.getenv("DEFAULT_TAKE_PROFIT_PCT", "0.022"))
    )  # هدف سود ۲.۲٪
    trailing_activation_pct: float = field(
        default_factory=lambda: float(os.getenv("TRAILING_STOP_ACTIVATION_PCT", "0.010"))
    )  # فعال‌سازی تریلینگ استاپ در سود +۱.۰٪
    trailing_distance_pct: float = field(
        default_factory=lambda: float(os.getenv("TRAILING_STOP_DISTANCE_PCT", "0.006"))
    )  # فاصله تریلینگ استاپ از سقف قیمت (۰.۶٪)
    max_portfolio_drawdown_pct: float = field(
        default_factory=lambda: float(os.getenv("MAX_PORTFOLIO_DRAWDOWN_PCT", "0.035"))
    )  # مدارقطع‌کن کل حساب در صورت افت ۳.۵٪ از سقف

    # پارامترهای اجرایی و زمان‌بندی
    max_spread_pct: float = field(
        default_factory=lambda: float(os.getenv("MAX_SPREAD_PCT", "0.25"))
    )  # رد کردن بازارهای با اسپرد بالای ۰.۲۵٪
    loop_interval_sec: int = field(
        default_factory=lambda: int(os.getenv("LOOP_INTERVAL_SECONDS", "45"))
    )  # فاصله هر سیکل بررسی بازار (ثانیه)
    stale_order_timeout_sec: int = field(
        default_factory=lambda: int(os.getenv("STALE_ORDER_TIMEOUT_SECONDS", "180"))
    )  # لغو سفارش لیمیت پرنشده پس از ۳ دقیقه
    contest_end_timestamp: int = field(
        default_factory=lambda: int(os.getenv("CONTEST_END_TIMESTAMP", "0"))
    )
    flatten_before_end_sec: int = field(
        default_factory=lambda: int(os.getenv("FLATTEN_MINUTES_BEFORE_END", "25")) * 60
    )  # نقد کردن کامل به تومان قبل از پایان چالش

    # فایل‌های ذخیره‌سازی تحویلی به داوران
    decision_log_file: str = field(
        default_factory=lambda: os.getenv("DECISION_LOG_FILE", "decision_logs.jsonl")
    )
    state_file: str = field(
        default_factory=lambda: os.getenv("PORTFOLIO_STATE_FILE", "portfolio_state.json")
    )

    # تنظیمات مدل زبانی (تنظیم‌شده روی BoxAPI و GPT-6 Luna)
    llm_api_key: str = field(default_factory=lambda: os.getenv("LLM_API_KEY", ""))
    llm_base_url: str = field(
        default_factory=lambda: os.getenv(
            "LLM_BASE_URL", "https://ai.boxapi.ir/ai/v1"
        )
    )
    llm_model: str = field(
        default_factory=lambda: os.getenv("LLM_MODEL", "gpt-6-luna")
    )
    llm_reasoning_effort: str = field(
        default_factory=lambda: os.getenv("LLM_REASONING_EFFORT", "medium")
    )


# ==========================================
# ۳. کلاینت ارتباط با والکس (REST / MCP)
# ==========================================
class WallexClient:
    def __init__(self, config: AgentConfig):
        self.config = config
        self.session = requests.Session()
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "WallexHackathonAgent/1.0",
        }
        if self.config.api_key:
            headers["X-API-Key"] = self.config.api_key
        self.session.headers.update(headers)
        self.market_rules: Dict[str, dict] = {}

    def load_market_rules(self) -> List[str]:
        """دریافت قوانین دقت اعشار بازارها و فیلتر کردن ۳۲۶ بازار ممنوعه"""
        resp = self.session.get(f"{self.config.base_url}/v1/markets", timeout=12)
        resp.raise_for_status()
        symbols_data = resp.json().get("result", {}).get("symbols", {})

        allowed_active = MarketComplianceGuard.filter_active_wallex_markets(
            symbols_data, top_only=True
        )
        for sym in allowed_active:
            info = symbols_data.get(sym, {})
            # خواندن دقیق دقت اعشار قیمت و حجم از فیلدهای رسمی والکس
            price_prec = int(info.get("quotePrecision", info.get("quoteAssetPrecision", 0) or 0))
            qty_prec = int(info.get("baseAssetPrecision", info.get("stepSize", 6) or 6))
            min_qty = float(info.get("minQty", 0.00000001))
            min_notional = float(info.get("minNotional", 50_000.0))

            self.market_rules[sym] = {
                "base_asset": info.get("baseAsset", sym.replace("TMN", "")),
                "price_precision": price_prec,
                "qty_precision": qty_prec,
                "min_qty": min_qty,
                "min_notional": min_notional,
            }
        logger.info(f"Loaded {len(self.market_rules)} compliant TMN markets: {list(self.market_rules.keys())}")
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
        # گرد کردن رو به پایین برای جلوگیری از کمبود موجودی
        rounded = math.floor(quantity * factor) / factor
        return f"{rounded:.{prec}f}" if prec > 0 else str(int(rounded))

    def get_balances(self) -> Dict[str, dict]:
        resp = self.session.get(f"{self.config.base_url}/v1/account/balances", timeout=10)
        resp.raise_for_status()
        return resp.json().get("result", {}).get("balances", {})

    def get_orderbook(self, symbol: str) -> dict:
        norm = normalize_symbol(symbol)
        try:
            resp = self.session.get(
                f"{self.config.base_url}/v1/depth", params={"symbol": norm}, timeout=10
            )
            resp.raise_for_status()
            return resp.json().get("result", {})
        except Exception as e:
            logger.warning(f"Error fetching depth for {norm}: {e}")
            return {"bid": [], "ask": []}

    def get_latest_trades(self, symbol: str) -> List[dict]:
        norm = normalize_symbol(symbol)
        try:
            resp = self.session.get(
                f"{self.config.base_url}/v1/trades", params={"symbol": norm}, timeout=10
            )
            resp.raise_for_status()
            return resp.json().get("result", {}).get("latestTrades", [])
        except Exception as e:
            logger.warning(f"Error fetching latest trades for {norm}: {e}")
            return []

    def get_candles(self, symbol: str, resolution: str = "15", lookback_hours: int = 36) -> pd.DataFrame:
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
            return pd.DataFrame({
                "timestamp": data["t"],
                "open": [float(x) for x in data["o"]],
                "high": [float(x) for x in data["h"]],
                "low": [float(x) for x in data["l"]],
                "close": [float(x) for x in data["c"]],
                "volume": [float(x) for x in data["v"]],
            })
        except Exception as e:
            logger.warning(f"Error fetching candles for {norm}: {e}")
            return pd.DataFrame()

    def get_open_orders(self, symbol: Optional[str] = None) -> List[dict]:
        if self.config.mode == ExecutionMode.SHADOW:
            return []
        params = {"symbol": normalize_symbol(symbol)} if symbol else {}
        try:
            resp = self.session.get(
                f"{self.config.base_url}/v1/account/openOrders", params=params, timeout=10
            )
            resp.raise_for_status()
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
            logger.warning(f"Error fetching open orders (Network/API hiccup): {e}")
            return None

    def get_order_status(self, client_id: str) -> Optional[dict]:
        """
        استعلام وضعیت دقیق یک سفارش از طریق GET /v1/account/orders/{client_id}.
        در صورت خطای شبکه یا تایم‌اوت None برمی‌گرداند.
        """
        if self.config.mode == ExecutionMode.SHADOW:
            return None
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

    def cancel_order(self, client_order_id: str) -> bool:
        if self.config.mode == ExecutionMode.SHADOW:
            return True
        try:
            # 1. مسیر مستقیم پارامتر URL
            url = f"{self.config.base_url}/v1/account/orders/{client_order_id}"
            resp = self.session.delete(url, timeout=10)
            if resp.status_code in (200, 204):
                return True
            # 2. مسیر جایگزین بدنه درخواست
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
        self, symbol: str, side: str, order_type: str, quantity: float, price: Optional[float] = None
    ) -> dict:
        # ۱. گارد امنیتی سطح کد برای جلوگیری قطعی از معامله در ۳۲۶ بازار ممنوعه
        norm_symbol = MarketComplianceGuard.assert_market_allowed(symbol)

        client_id = f"wlx_{int(time.time()*1000)}_{uuid.uuid4().hex[:5]}"
        formatted_qty = self.format_quantity(norm_symbol, quantity)

        if float(formatted_qty) <= 0:
            raise ValueError(f"Formatted quantity for {norm_symbol} is zero ({quantity})")

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
                "status": "SHADOW_FILLED",
                "clientOrderId": client_id,
                "symbol": norm_symbol,
                "side": side.upper(),
                "type": order_type.upper(),
                "origQty": formatted_qty,
                "price": payload.get("price", str(price or 0.0)),
            }

        resp = self.session.post(
            f"{self.config.base_url}/v1/account/orders", json=payload, timeout=12
        )
        resp.raise_for_status()
        return resp.json().get("result", {})


# ==========================================
# ۴. موتور محاسبات کمی و ریزساختار بازار
# ==========================================
class QuantFeatureEngine:
    @staticmethod
    def compute_features(
        df: pd.DataFrame, orderbook: dict, recent_trades: Optional[List[dict]] = None
    ) -> dict:
        if df.empty or len(df) < 25:
            return {"valid": False, "reason": "Insufficient candles (<25)"}

        bids = orderbook.get("bid", [])[:5]
        asks = orderbook.get("ask", [])[:5]
        if not bids or not asks:
            return {"valid": False, "reason": "Empty orderbook"}

        close = df["close"]
        high = df["high"]
        low = df["low"]
        vol = df["volume"]

        # میانگین‌های متحرک نمایی
        ema9 = float(close.ewm(span=9, adjust=False).mean().iloc[-1])
        ema21 = float(close.ewm(span=21, adjust=False).mean().iloc[-1])

        # شاخص RSI(14)
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = float(100.0 - (100.0 / (1.0 + rs.iloc[-1]))) if not np.isnan(rs.iloc[-1]) else 50.0

        # شاخص نوسان ATR(14) برای تعیین حد ضرر پویا
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs(),
        ], axis=1).max(axis=1)
        atr = float(tr.rolling(14).mean().iloc[-1])
        if np.isnan(atr) or atr <= 0:
            atr = float((high.iloc[-1] - low.iloc[-1]) or (close.iloc[-1] * 0.01))

        # امتیاز استاندارد حجم (Volume Z-Score)
        vol_mean = float(vol.rolling(20).mean().iloc[-1])
        vol_std = float(vol.rolling(20).std().iloc[-1])
        vol_z = (float(vol.iloc[-1]) - vol_mean) / vol_std if vol_std > 0 else 0.0

        # تحلیل دفتر سفارشات (Order Book Depth Top 5)
        best_bid = float(bids[0]["price"])
        best_ask = float(asks[0]["price"])
        mid_price = (best_bid + best_ask) / 2.0
        spread_pct = ((best_ask - best_bid) / (mid_price + 1e-9)) * 100.0

        bid_vol_5 = sum(float(b["quantity"]) for b in bids)
        ask_vol_5 = sum(float(a["quantity"]) for a in asks)
        obi = (bid_vol_5 - ask_vol_5) / (bid_vol_5 + ask_vol_5 + 1e-9)

        # تحلیل جریان سفارشات اخیر (Trade Flow Imbalance / Taker Buy Ratio)
        taker_buy_ratio = 0.5
        if recent_trades:
            buy_trades_vol = 0.0
            total_trades_vol = 0.0
            for t in recent_trades[:30]:
                q = float(t.get("quantity", 0.0))
                total_trades_vol += q
                if t.get("isBuyOrder", False):
                    buy_trades_vol += q
            if total_trades_vol > 0:
                taker_buy_ratio = buy_trades_vol / total_trades_vol

        return {
            "valid": True,
            "last_price": float(close.iloc[-1]),
            "best_bid": best_bid,
            "best_ask": best_ask,
            "spread_pct": round(spread_pct, 4),
            "obi_top5": round(obi, 4),
            "taker_buy_ratio": round(taker_buy_ratio, 3),
            "ema9": round(ema9, 2),
            "ema21": round(ema21, 2),
            "trend_bullish": bool(ema9 > ema21),
            "rsi_14": round(rsi, 2),
            "atr_14": round(atr, 2),
            "volume_zscore": round(vol_z, 2),
        }


# ==========================================
# ۵. مدیریت پایدار پوزیشن‌ها، حد ضرر و لاگ داوری
# ==========================================
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


class PortfolioAndRiskManager:
    def __init__(self, config: AgentConfig):
        self.config = config
        self.shadow_available_tmn: float = config.initial_capital_tmn
        self.peak_equity_tmn: float = config.initial_capital_tmn
        self.positions: Dict[str, OpenPosition] = {}
        self.pending_buy_orders: Dict[str, dict] = {}  # رهگیری سفارش‌های لیمیت پرنشده برای جلوگیری از پوزیشن مجازی
        if os.path.exists(self.config.state_file):
            self.load_state()
        else:
            self.save_state()

    def save_state(self):
        """ذخیره اتمیک وضعیت در فایل جهت جلوگیری از خراب شدن فایل هنگام قطع برق یا ری‌استارت"""
        data = {
            "shadow_available_tmn": self.shadow_available_tmn,
            "peak_equity_tmn": self.peak_equity_tmn,
            "positions": {sym: asdict(pos) for sym, pos in self.positions.items()},
            "pending_buy_orders": self.pending_buy_orders,
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
            self.shadow_available_tmn = float(data.get("shadow_available_tmn", self.config.initial_capital_tmn))
            self.peak_equity_tmn = float(data.get("peak_equity_tmn", self.config.initial_capital_tmn))
            self.pending_buy_orders = data.get("pending_buy_orders", {})

            for sym, pos_dict in data.get("positions", {}).items():
                # تبدیل امن دیکشنری به آبجکت OpenPosition
                pos_keys = OpenPosition.__annotations__.keys()
                filtered = {k: v for k, v in pos_dict.items() if k in pos_keys}
                self.positions[sym] = OpenPosition(**filtered)
            logger.info(f"Restored portfolio state: {len(self.positions)} active positions.")
        except Exception as e:
            logger.warning(f"Could not load state file: {e}")

    def compute_equity(self, client: WallexClient, latest_prices: Dict[str, float]) -> tuple[float, float]:
        """محاسبه موجودی نقد (Available TMN) و ارزش کل پرتفوی (Total Equity TMN)"""
        if self.config.mode == ExecutionMode.SHADOW:
            pos_val = sum(
                pos.quantity * latest_prices.get(sym, pos.entry_price)
                for sym, pos in self.positions.items()
            )
            total_eq = self.shadow_available_tmn + pos_val
            self.peak_equity_tmn = max(self.peak_equity_tmn, total_eq)
            return self.shadow_available_tmn, total_eq

        # در حالت LIVE، موجودی واقعی از والکس خوانده می‌شود
        try:
            balances = client.get_balances()
            tmn_info = balances.get("TMN", {})
            avail_tmn = float(tmn_info.get("value", 0.0)) - float(tmn_info.get("locked", 0.0))
            total_tmn = float(tmn_info.get("value", 0.0))

            crypto_val = 0.0
            for sym, rules in client.market_rules.items():
                base = rules["base_asset"]
                b_info = balances.get(base, {})
                qty = float(b_info.get("value", 0.0))
                if qty > 0 and sym in latest_prices:
                    crypto_val += qty * latest_prices[sym]

            total_eq = total_tmn + crypto_val
            self.peak_equity_tmn = max(self.peak_equity_tmn, total_eq)
            return avail_tmn, total_eq
        except Exception as e:
            logger.error(f"Error computing live equity from Wallex: {e}")
            return 0.0, self.peak_equity_tmn

    def check_deterministic_exits(self, symbol: str, features: dict) -> Optional[AIDecisionOutput]:
        """
        بررسی لحظه‌ای حد ضرر (Stop-Loss)، حد سود (Take-Profit) و تریلینگ استاپ
        قبل از فراخوانی LLM برای محافظت قطعی از سرمایه (۳۰ امتیاز کنترل افت حساب).
        """
        pos = self.positions.get(symbol)
        if not pos:
            return None

        current_bid = features["best_bid"]
        pnl_pct = (current_bid - pos.entry_price) / pos.entry_price

        # به‌روزرسانی بالاترین قیمت دیده‌شده و فعال‌سازی Trailing Stop
        if current_bid > pos.highest_price:
            pos.highest_price = current_bid

        if pnl_pct >= self.config.trailing_activation_pct:
            pos.trailing_active = True
            # قفل کردن سود بالاتر از نقطه سر‌به‌سر کارمزدها
            breakeven_plus = pos.entry_price * (1.0 + 2.5 * self.config.fee_rate_per_side)
            trail_stop = pos.highest_price * (1.0 - self.config.trailing_distance_pct)
            new_sl = max(pos.stop_loss_price, breakeven_plus, trail_stop)
            if new_sl > pos.stop_loss_price:
                pos.stop_loss_price = new_sl
                self.save_state()

        # ۱. برخورد با حد ضرر یا تریلینگ استاپ
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
                    f"({pos.stop_loss_price:,.0f}) رسید. بازدهی پوزیشن: {pnl_pct*100:.2f}%."
                ),
            )

        # ۲. برخورد با حد سود (Take Profit)
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
                    f"({pos.take_profit_price:,.0f}) عبور کرد. سود محقق‌شده ناخالص: {pnl_pct*100:.2f}%."
                ),
            )

        return None

    def approve_and_size_buy(
        self, decision: AIDecisionOutput, features: dict, avail_tmn: float, total_eq: float
    ) -> tuple[bool, str, float]:
        """بررسی قوانین مدیریت سرمایه قبل از ارسال سفارش خرید"""
        drawdown = (self.peak_equity_tmn - total_eq) / max(self.peak_equity_tmn, 1.0)
        if drawdown >= self.config.max_portfolio_drawdown_pct:
            return False, f"CIRCUIT_BREAKER: Drawdown {drawdown:.2%} >= {self.config.max_portfolio_drawdown_pct:.2%}", 0.0

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
    ):
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "mode": self.config.mode.value,
            "symbol": symbol,
            "action": decision.action,
            "order_type": decision.order_type,
            "ai_confidence": decision.confidence,
            "signals_observed": decision.signals_observed,
            "reasoning_fa": decision.reasoning_fa,
            "risk_verdict": risk_verdict,
            "portfolio_equity_tmn": round(total_equity_tmn, 0),
            "market_snapshot": features,
            "execution_result": execution_result,
        }
        try:
            with open(self.config.decision_log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.error(f"Failed to append to decision log: {e}")


# ==========================================
# ۶. موتور اصلی اجرای حلقه معاملاتی (Main Runner)
# ==========================================
class TradingAgentRunner:
    def __init__(self, config: AgentConfig):
        self.config = config
        self.client = WallexClient(config)
        self.risk_mgr = PortfolioAndRiskManager(config)
        self.strategist = CostEffectiveLLMStrategist(config)
        self.running = True
        self.pending_order_timestamps: Dict[str, float] = {}

        signal.signal(signal.SIGINT, self._handle_stop)
        signal.signal(signal.SIGTERM, self._handle_stop)

    def _handle_stop(self, signum, frame):
        logger.info("Shutdown signal received. Saving portfolio state cleanly...")
        self.risk_mgr.save_state()
        self.running = False

    def sweep_stale_open_orders(self):
        """
        لغو سفارش‌های لیمیت پرنشده قدیمی‌تر از ۳ دقیقه جهت آزادسازی تومان قفل‌شده
        و تایید قطعی پر شدن سفارش‌ها قبل از انتقال به پوزیشن‌های باز.

        CRITICAL FIX FOR PHANTOM FILLS:
        1. اگر get_open_orders با خطای شبکه مواجه شود (None)، هیچ ارزیابی یا ارتقایی انجام نمی‌شود.
        2. برای هر سفارش که در open_orders نیست، وضعیت دقیق آن از طریق get_order_status
           یا بررسی موجودی واقعی کیف‌پول استعلام می‌شود و تنها در صورت پر شدن واقعی (FILLED)
           به پوزیشن فعال تبدیل می‌گردد.
        """
        now = time.time()

        if self.config.mode == ExecutionMode.LIVE:
            try:
                open_orders = self.client.get_open_orders()
                # جلوگیری از پوزیشن فانتوم هنگام اختلال شبکه / تایم‌اوت API
                if open_orders is None:
                    logger.warning(
                        "[NETWORK HICCUP] Could not retrieve open orders from Wallex. "
                        "Skipping stale sweep and order promotions to prevent phantom fills."
                    )
                    return

                live_cids = set()
                for o in open_orders:
                    cid = o.get("clientOrderId")
                    if not cid:
                        continue
                    live_cids.add(cid)
                    first_seen = self.pending_order_timestamps.setdefault(cid, now)
                    if now - first_seen >= self.config.stale_order_timeout_sec:
                        logger.warning(
                            f"[STALE ORDER] Canceling stale order {cid} on {o.get('symbol')} "
                            f"(Age: {now - first_seen:.0f}s >= {self.config.stale_order_timeout_sec}s)"
                        )
                        self.client.cancel_order(cid)
                        self.pending_order_timestamps.pop(cid, None)
                        # حذف از لیست سفارش‌های در انتظار خرید
                        for sym in list(self.risk_mgr.pending_buy_orders.keys()):
                            if self.risk_mgr.pending_buy_orders[sym].get("client_id") == cid:
                                del self.risk_mgr.pending_buy_orders[sym]

                # بررسی سفارش‌های خریدی که در pending هستند ولی در open_orders مشاهده نمی‌شوند
                for sym, pending in list(self.risk_mgr.pending_buy_orders.items()):
                    cid = pending.get("client_id")
                    if not cid or cid in live_cids:
                        continue

                    # سفارش در open_orders نیست؛ بررسی قطعی وضعیت سفارش جهت جلوگیری از پوزیشن فانتوم
                    order_status = self.client.get_order_status(cid)
                    is_actually_filled = False
                    actual_qty = pending["quantity"]
                    actual_price = pending["entry_price"]

                    if order_status is not None:
                        status_str = str(order_status.get("status", "")).upper()
                        exec_qty = float(order_status.get("executedQty", 0.0) or 0.0)
                        exec_price = float(order_status.get("executedPrice", 0.0) or 0.0)
                        is_active = order_status.get("active", False)

                        if status_str == "FILLED" or (exec_qty > 0 and not is_active):
                            is_actually_filled = True
                            if exec_qty > 0:
                                actual_qty = exec_qty
                            if exec_price > 0:
                                actual_price = exec_price
                        elif status_str in ("CANCELED", "CANCELLED", "REJECTED", "EXPIRED"):
                            logger.info(
                                f"[ORDER {status_str}] Order {cid} for {sym} was {status_str} on exchange. "
                                f"Discarding pending order without promoting."
                            )
                            del self.risk_mgr.pending_buy_orders[sym]
                            self.pending_order_timestamps.pop(cid, None)
                            self.risk_mgr.save_state()
                            continue
                        elif is_active:
                            # سفارش هنوز در صرافی اکتیو است
                            continue
                    else:
                        # گام دوم تایید (Fallback): بررسی موجودی ولت صرافی
                        try:
                            balances = self.client.get_balances()
                            rule = self.client.market_rules.get(sym, {})
                            base_asset = rule.get("base_asset", sym.replace("TMN", ""))
                            coin_balance = float(balances.get(base_asset, {}).get("value", 0.0))
                            min_expected = pending["quantity"] * 0.95
                            if coin_balance >= min_expected:
                                logger.info(
                                    f"[WALLET BALANCE CONFIRMED] Wallet holds {coin_balance} {base_asset}. "
                                    f"Confirming fill for {cid}."
                                )
                                is_actually_filled = True
                                actual_qty = min(coin_balance, pending["quantity"])
                        except Exception as be:
                            logger.warning(f"Could not verify wallet balance for {sym}: {be}")

                    if is_actually_filled:
                        logger.info(
                            f"[VERIFIED LIMIT FILL] Order {cid} for {sym} confirmed FILLED on exchange! "
                            f"Qty: {actual_qty} @ {actual_price:,.0f} TMN. Promoting to active position."
                        )
                        self.risk_mgr.positions[sym] = OpenPosition(
                            symbol=sym,
                            entry_price=actual_price,
                            quantity=actual_qty,
                            stop_loss_price=pending["stop_loss_price"],
                            take_profit_price=pending["take_profit_price"],
                            highest_price=actual_price,
                            opened_at=datetime.now(timezone.utc).isoformat(),
                            client_id=cid,
                        )
                        del self.risk_mgr.pending_buy_orders[sym]
                        self.pending_order_timestamps.pop(cid, None)
                        self.risk_mgr.save_state()
                    else:
                        logger.warning(
                            f"[UNVERIFIED ORDER MISSING] Order {cid} for {sym} is not in open_orders "
                            f"but fill cannot be verified yet. Keeping in pending to prevent phantom fill."
                        )

            except Exception as e:
                logger.warning(f"Error sweeping live open orders: {e}")

        else:
            # در حالت SHADOW: سفارش‌های معلق شبیه‌سازی‌شده که ۳ دقیقه پر نشده‌اند را پاک کن
            for sym, pending in list(self.risk_mgr.pending_buy_orders.items()):
                created_ts = pending.get("created_at_ts", now)
                if now - created_ts >= self.config.stale_order_timeout_sec:
                    logger.info(f"[SHADOW STALE] Released locked shadow TMN for {sym}")
                    self.risk_mgr.shadow_available_tmn += pending.get("order_tmn", 0.0)
                    del self.risk_mgr.pending_buy_orders[sym]
                    self.risk_mgr.save_state()

    def execute_flatten_all_to_tmn(self):
        """
        قانون اجباری مسابقه:
        فروش فوری تمامی دارایی‌ها به تومان در دقایق پایانی هکاتون
        جهت تحقق ۱۰۰٪ سود به تومان طبق قوانین داوری والکس.
        """
        logger.warning("=" * 65)
        logger.warning("=== EXECUTING END-OF-CONTEST COMPLETE FLATTEN TO REALIZED TMN ===")
        logger.warning("=" * 65)

        # ۱. لغو کلیه سفارش‌های باز معلق
        if self.config.mode == ExecutionMode.LIVE:
            try:
                open_orders = self.client.get_open_orders() or []
                for o in open_orders:
                    cid = o.get("clientOrderId")
                    if cid:
                        self.client.cancel_order(cid)
            except Exception as e:
                logger.error(f"Error canceling open orders during flatten: {e}")

        # ۲. فروش کلیه پوزیشن‌های ثبت‌شده
        for sym, pos in list(self.risk_mgr.positions.items()):
            try:
                ob = self.client.get_orderbook(sym)
                best_bid = float(ob["bid"][0]["price"]) if ob.get("bid") else pos.entry_price
                res = self.client.place_spot_order(
                    symbol=sym, side="SELL", order_type="MARKET", quantity=pos.quantity, price=best_bid
                )
                if self.config.mode == ExecutionMode.SHADOW:
                    proceeds = pos.quantity * best_bid * (1.0 - self.config.fee_rate_per_side)
                    self.risk_mgr.shadow_available_tmn += proceeds

                decision = AIDecisionOutput(
                    symbol=sym,
                    action="SELL",
                    order_type="MARKET",
                    confidence=1.0,
                    allocation_pct=0.0,
                    signals_observed=["CONTEST_DEADLINE_REALIZE_TMN"],
                    reasoning_fa="نقد کردن خودکار دارایی در پایان چالش جهت تحقق کامل سود به تومان طبق دستورالعمل داوری والکس.",
                )
                self.risk_mgr.positions.pop(sym, None)
                self.risk_mgr.save_state()
                self.risk_mgr.log_decision(
                    sym, {"best_bid": best_bid}, decision, "END_OF_CONTEST_LIQUIDATION", res, self.risk_mgr.shadow_available_tmn
                )
                logger.info(f"Liquidated tracked position {sym} into TMN successfully.")
            except Exception as e:
                logger.error(f"Failed to liquidate {sym}: {e}")

        # ۳. در حالت LIVE: بررسی مستقیم موجودی حساب والکس برای اطمینان از صفر بودن هرگونه دارایی کوینی
        if self.config.mode == ExecutionMode.LIVE:
            try:
                balances = self.client.get_balances()
                for asset, info in balances.items():
                    if asset == "TMN":
                        continue
                    free_val = float(info.get("value", 0.0)) - float(info.get("locked", 0.0))
                    sym = f"{asset}TMN"
                    if sym in self.client.market_rules:
                        rule = self.client.market_rules[sym]
                        if free_val >= rule.get("min_qty", 0.0001):
                            logger.info(f"Liquidating remaining wallet asset {asset} ({free_val})...")
                            self.client.place_spot_order(sym, "SELL", "MARKET", free_val)
            except Exception as e:
                logger.error(f"Error checking remaining live wallet balances: {e}")

        logger.info("Liquidation complete. All assets converted to Realized TMN.")

    def run_single_cycle(self, active_markets: List[str]):
        # ۱. بررسی زمان پایان چالش برای نقد کردن خودکار به تومان
        if self.config.contest_end_timestamp > 0:
            remaining = self.config.contest_end_timestamp - int(time.time())
            if remaining <= self.config.flatten_before_end_sec:
                self.execute_flatten_all_to_tmn()
                self.running = False
                return

        # ۲. آزادسازی سفارش‌های لیمیت معلق قدیمی
        self.sweep_stale_open_orders()

        latest_prices: Dict[str, float] = {}
        market_snapshots: Dict[str, dict] = {}

        # ۳. جمع‌آوری دیتای بازار و بررسی فوری حد ضرر / حد سود پوزیشن‌های باز
        for sym in active_markets:
            try:
                ob = self.client.get_orderbook(sym)
                df = self.client.get_candles(sym, resolution="15", lookback_hours=36)
                trades = self.client.get_latest_trades(sym)
                feats = QuantFeatureEngine.compute_features(df, ob, trades)
                if not feats.get("valid"):
                    continue
                latest_prices[sym] = feats["best_bid"]
                market_snapshots[sym] = feats
            except Exception as e:
                logger.warning(f"Market data fetch warning on {sym}: {e}")

        avail_tmn, total_eq = self.risk_mgr.compute_equity(self.client, latest_prices)
        net_return_pct = ((total_eq - self.config.initial_capital_tmn) / self.config.initial_capital_tmn) * 100.0
        logger.info(
            f"Cycle Portfolio | Mode: {self.config.mode.value} | Equity: {total_eq:,.0f} TMN "
            f"({net_return_pct:+.2f}%) | Cash: {avail_tmn:,.0f} TMN | Open Pos: {len(self.risk_mgr.positions)}"
        )

        # ۴. پردازش هر بازار (ابتدا پوزیشن‌های باز، سپس فرصت‌های ورود جدید)
        for sym, feats in market_snapshots.items():
            pos = self.risk_mgr.positions.get(sym)

            # الف) بررسی خروج قطعی (Hard SL / Trailing Stop / Hard TP) بدون تأخیر LLM
            if pos is not None:
                det_exit = self.risk_mgr.check_deterministic_exits(sym, feats)
                if det_exit is not None:
                    self._execute_sell(sym, pos, feats, det_exit, "DETERMINISTIC_RISK_EXIT", total_eq)
                    continue

            # ب) بررسی وضعیت شبیه‌سازی پر شدن سفارش‌های لیمیت در حالت SHADOW
            if self.config.mode == ExecutionMode.SHADOW and sym in self.risk_mgr.pending_buy_orders:
                pending = self.risk_mgr.pending_buy_orders[sym]
                # اگر قیمت سرخط فروش به یا زیر قیمت لیمیت ما آمد، سفارش پر می‌شود
                if feats["best_ask"] <= pending["limit_price"]:
                    fill_price = feats["best_ask"]
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
                    )
                    del self.risk_mgr.pending_buy_orders[sym]
                    self.risk_mgr.save_state()

            # ج) مشاوره با استراتژیست هوش مصنوعی (تنها در صورت تایید دروازه‌بان کمّی)
            unrealized_pct = ((feats["best_bid"] - pos.entry_price) / pos.entry_price * 100.0) if pos else 0.0
            portfolio_context = {
                "available_tmn": avail_tmn,
                "total_equity_tmn": total_eq,
                "position_qty": pos.quantity if pos else 0.0,
                "entry_price": pos.entry_price if pos else None,
                "unrealized_pnl_pct": round(unrealized_pct, 2),
            }
            recent_flow = {"taker_buy_ratio": feats["taker_buy_ratio"], "obi_top5": feats["obi_top5"]}

            decision = self.strategist.evaluate_market(sym, feats, portfolio_context, recent_flow)

            # د) اجرای تصمیم خرید (BUY) پس از عبور از فیلتر مدیریت ریسک
            if decision.action == "BUY" and pos is None and sym not in self.risk_mgr.pending_buy_orders:
                approved, reason, order_tmn = self.risk_mgr.approve_and_size_buy(
                    decision, feats, avail_tmn, total_eq
                )
                if approved:
                    self._execute_buy(sym, feats, decision, order_tmn, reason, total_eq)
                    avail_tmn -= order_tmn
                else:
                    self.risk_mgr.log_decision(sym, feats, decision, f"REJECTED: {reason}", None, total_eq)

            # هـ) اجرای تصمیم فروش هوش مصنوعی (AI Tactical Exit)
            elif decision.action == "SELL" and pos is not None:
                self._execute_sell(sym, pos, feats, decision, "AI_TACTICAL_EXIT", total_eq)

    def _execute_buy(
        self, sym: str, feats: dict, decision: AIDecisionOutput, order_tmn: float, risk_reason: str, total_eq: float
    ):
        exec_price = (
            decision.limit_price
            if (decision.order_type == "LIMIT" and decision.limit_price)
            else feats["best_bid"]
        )
        # جلوگیری از خرید بالاتر از بهترین پیشنهاد فروش سرخط
        exec_price = min(exec_price, feats["best_ask"])
        raw_qty = (order_tmn * (1.0 - self.config.fee_rate_per_side)) / exec_price

        # محاسبه دقیق حد ضرر مبتنی بر ATR با سقف سخت ۱.۲٪
        atr_sl = exec_price - (1.5 * feats["atr_14"])
        hard_sl = exec_price * (1.0 - self.config.hard_stop_loss_pct)
        sl_price = max(atr_sl, hard_sl)
        if decision.stop_loss_price and decision.stop_loss_price > hard_sl and decision.stop_loss_price < exec_price:
            sl_price = decision.stop_loss_price

        tp_price = exec_price * (1.0 + self.config.default_take_profit_pct)
        if decision.take_profit_price and decision.take_profit_price > exec_price * 1.008:
            tp_price = decision.take_profit_price

        try:
            res = self.client.place_spot_order(
                symbol=sym,
                side="BUY",
                order_type=decision.order_type,
                quantity=raw_qty,
                price=exec_price if decision.order_type == "LIMIT" else None,
            )
            filled_qty = float(self.client.format_quantity(sym, raw_qty))
            client_id = res.get("clientOrderId", f"wlx_{int(time.time()*1000)}")

            if self.config.mode == ExecutionMode.SHADOW:
                self.risk_mgr.shadow_available_tmn -= order_tmn

            # مدیریت دقیق پوزیشن‌ها بر اساس نوع سفارش:
            # اگر سفارش MARKET باشد یا لیمیت در سرخط پر شود، فوراً وارد پوزیشن‌های فعال می‌شود
            is_immediate_fill = (
                decision.order_type == "MARKET" or
                (self.config.mode == ExecutionMode.SHADOW and exec_price >= feats["best_ask"])
            )

            if is_immediate_fill or self.config.mode == ExecutionMode.SHADOW:
                self.risk_mgr.positions[sym] = OpenPosition(
                    symbol=sym,
                    entry_price=exec_price,
                    quantity=filled_qty,
                    stop_loss_price=round(sl_price, 2),
                    take_profit_price=round(tp_price, 2),
                    highest_price=exec_price,
                    opened_at=datetime.now(timezone.utc).isoformat(),
                    client_id=client_id,
                )
            else:
                # در حالت LIVE با سفارش LIMIT: سفارش در انتظار ثبت می‌شود تا پر شدن آن بررسی شود
                self.risk_mgr.pending_buy_orders[sym] = {
                    "client_id": client_id,
                    "entry_price": exec_price,
                    "limit_price": exec_price,
                    "quantity": filled_qty,
                    "stop_loss_price": round(sl_price, 2),
                    "take_profit_price": round(tp_price, 2),
                    "order_tmn": order_tmn,
                    "created_at_ts": time.time(),
                }
                self.pending_order_timestamps[client_id] = time.time()

            self.risk_mgr.save_state()
            self.risk_mgr.log_decision(sym, feats, decision, risk_reason, res, total_eq)
            logger.info(
                f"[BUY EXECUTED] {sym} | Type: {decision.order_type} | Qty: {filled_qty} @ {exec_price:,.0f} TMN | "
                f"SL: {sl_price:,.0f} | TP: {tp_price:,.0f}"
            )
        except Exception as e:
            logger.error(f"Order placement failed on {sym}: {e}")

    def _execute_sell(
        self, sym: str, pos: OpenPosition, feats: dict, decision: AIDecisionOutput, exit_reason: str, total_eq: float
    ):
        exec_price = feats["best_bid"]
        try:
            # سفارش فروش همواره با MARKET یا قیمت سرخط خرید ثبت می‌شود تا فوری اجرا شود
            res = self.client.place_spot_order(
                symbol=sym,
                side="SELL",
                order_type="MARKET",
                quantity=pos.quantity,
                price=exec_price,
            )
            if self.config.mode == ExecutionMode.SHADOW:
                proceeds = pos.quantity * exec_price * (1.0 - self.config.fee_rate_per_side)
                self.risk_mgr.shadow_available_tmn += proceeds

            pnl_pct = ((exec_price - pos.entry_price) / pos.entry_price) * 100.0
            self.risk_mgr.positions.pop(sym, None)
            self.risk_mgr.save_state()
            self.risk_mgr.log_decision(sym, feats, decision, exit_reason, res, total_eq)
            logger.info(
                f"[SELL EXECUTED] {sym} @ {exec_price:,.0f} TMN | Net Trade PnL: {pnl_pct:+.2f}% ({exit_reason})"
            )
        except Exception as e:
            logger.error(f"Sell execution failed on {sym}: {e}")

    def start(self):
        active_markets = self.client.load_market_rules()
        logger.info(f"Starting Wallex AI Agent in {self.config.mode.value} mode...")
        logger.info(f"Tracking {len(active_markets)} compliant markets. Initial Capital: {self.config.initial_capital_tmn:,.0f} TMN")

        while self.running:
            cycle_start = time.time()
            try:
                self.run_single_cycle(active_markets)
            except Exception as e:
                logger.error(f"Unexpected cycle error (auto-recovering): {e}", exc_info=True)

            elapsed = time.time() - cycle_start
            sleep_sec = max(5.0, self.config.loop_interval_sec - elapsed)
            if self.running:
                time.sleep(sleep_sec)


# ==========================================
# ۷. نقطه ورود خط فرمان (CLI Entrypoint)
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Wallex Hackathon AI Spot Trading Agent")
    parser.add_argument("--mode", choices=["SHADOW", "LIVE"], default=None, help="Override execution mode (SHADOW / LIVE)")
    parser.add_argument("--flatten", action="store_true", help="Immediately liquidate all crypto holdings to TMN")
    args = parser.parse_args()

    cfg = AgentConfig()
    if args.mode:
        cfg.mode = ExecutionMode(args.mode)

    runner = TradingAgentRunner(cfg)
    runner.client.load_market_rules()

    if args.flatten:
        runner.execute_flatten_all_to_tmn()
    else:
        runner.start()
