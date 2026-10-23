# llm_strategist.py
"""
Cost-Effective LLM Strategist with Quant Gatekeeper & Cooldown Cache.
Fixes reasoning_effort token starvation, strips markdown fences, filters out dead/flat markets,
and logs structured JSON trading decisions with Persian audit reasoning for hackathon judges.
"""

import os
import re
import time
import json
import logging
from typing import Optional, Dict
from openai import OpenAI
from models import AIDecisionOutput
from config import AgentConfig
from market_compliance import MarketComplianceGuard

logger = logging.getLogger("WallexAgent")

COMPACT_SYSTEM_PROMPT = """You are an institutional Algorithmic Spot Trading Agent in the Wallex Hackathon (5,000,000 TMN capital).
Your mission: Maximize realized TMN return over the 36-hour contest while keeping Maximum Drawdown strictly under 3.0%.
You must output ONLY valid JSON matching this schema:
{
  "symbol": "string",
  "action": "BUY" | "SELL" | "HOLD",
  "order_type": "LIMIT" | "MARKET",
  "confidence": float (0.0 to 1.0),
  "allocation_pct": float (0.0 to 0.22),
  "limit_price": float or null,
  "stop_loss_price": float or null,
  "take_profit_price": float or null,
  "signals_observed": ["string"],
  "reasoning_fa": "توضیح دقیق فنی به فارسی با ذکر اعداد واقعی RSI, EMA, OBI, ATR%, Spread, Stop-Loss برای داوران"
}

DECISION LOGIC & CONTEST RULES:
1. FEE & VOLATILITY GUARD: Wallex Credit Account charges 0.25% Maker and 0.30% Taker (~0.55%-0.60% round-trip). Never BUY if spread_pct > 0.55% or atr_pct < 0.22% (insufficient volatility to cover fees).
2. BUY requires confluence of independent signals:
   - Setup A (Trend Pullback): macro_uptrend is true, trend_bullish is true (EMA9 > EMA21), RSI_14 between 38 and 62, and (obi_top5 >= +0.08 or taker_buy_ratio >= 0.52).
   - Setup B (Orderbook Buyer Absorption): strong buyer depth (obi_top5 >= +0.25) and aggressive buying (taker_buy_ratio >= 0.58) with positive momentum.
   - Setup C (Momentum Breakout): macro_uptrend is true, volume_zscore >= 1.2 or roc_3_pct >= 0.35%, obi_top5 >= +0.10, and RSI_14 < 68.
3. Order Pricing: Prefer LIMIT order at best_bid (for BUY) to capture Maker fee (0.25%), or MARKET only if volume_zscore > 2.0 and obi_top5 > 0.30.
4. Risk Bounds: Set stop_loss_price = max(last_price - 1.8 * atr_14, last_price * (1.0 - 0.014)) (capped at -1.4% to protect capital). Set take_profit_price = max(last_price + 2.8 * atr_14, last_price * 1.028) (target at least +2.8% to guarantee >2:1 reward-to-risk over fees).
5. Active Positions: DO NOT panic-sell at minor losses or intraday noise (0.55% fee friction will bleed equity). Stop-Loss (-1.4%) and Trailing-Stop (+1.5% activation) are handled deterministically. Recommend SELL only if:
   - The position is IN NET PROFIT (unrealized_pnl_pct >= +1.2%) and showing overbought exhaustion (RSI_14 > 72 with weakening buy ratio) to lock in gains before a dip.
   - The position is stagnant (>2 hours) and macro trend broke down (price < EMA50).
   Otherwise, output action: "HOLD" to allow the trade breathing room to reach the Take-Profit target (+2.8%).
6. Always write a rich, specific Persian justification in 'reasoning_fa' citing exact indicator values."""


class CostEffectiveLLMStrategist:
    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ):
        self.config = config or AgentConfig()
        self.api_key = api_key or self.config.llm_api_key or os.getenv("LLM_API_KEY", "")
        self.base_url = base_url or self.config.llm_base_url or os.getenv("LLM_BASE_URL", "https://ai.boxapi.ir/ai/v1")
        self.model = model or self.config.llm_model or os.getenv("LLM_MODEL", "gpt-6-luna")
        self.reasoning_effort = (
            self.config.llm_reasoning_effort or os.getenv("LLM_REASONING_EFFORT", "medium")
        )

        if self.api_key:
            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        else:
            self.client = None
            logger.warning("No LLM_API_KEY provided. Agent will use deterministic quant fallback mode.")

        self.total_llm_calls = 0
        # Tracks last HOLD evaluation per symbol to prevent spamming tokens on the same candle
        self._hold_cooldown_cache: Dict[str, tuple[float, float]] = {}

    @staticmethod
    def _clean_json_text(raw_text: str) -> str:
        """Strips markdown ```json fences or leading/trailing prose around JSON object."""
        text = raw_text.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return text[start : end + 1]
        return text

    def should_wake_llm(
        self, symbol: str, features: dict, has_open_position: bool, portfolio_state: Optional[dict] = None
    ) -> tuple[bool, str]:
        if not features.get("valid", False):
            return False, features.get("reason", "INVALID_DATA")

        # 1. Active Position Management: Let deterministic SL/TP/Trailing work!
        # NEVER wake LLM to cut minor pullbacks (<20m old or negative PnL above SL). Fee friction (0.55%) bleeds equity.
        if has_open_position:
            unrealized = portfolio_state.get("unrealized_pnl_pct", 0.0) if portfolio_state else 0.0
            time_held_sec = portfolio_state.get("time_held_sec", 0.0) if portfolio_state else 0.0

            # Profit-taking opportunity: lock in green profits (>+1.2%) on overbought exhaustion or heavy ask resistance
            if unrealized >= 1.2 and (features.get("rsi_14", 50.0) > 72.0 or features.get("obi_top5", 0.0) < -0.30):
                return True, f"ACTIVE_POSITION_PROFIT_TAKE_OPPORTUNITY_PNL_{unrealized:+.2f}%"

            # Stagnant breakdown: held for >2 hours and macro trend broke down (price < EMA50)
            if time_held_sec > 7200 and not features.get("macro_uptrend", True) and unrealized < -0.4:
                return True, f"STAGNANT_POSITION_MACRO_BREAKDOWN_{time_held_sec/60:.0f}M"

            return False, "POSITION_PROTECTED_BY_DETERMINISTIC_SL_TP"

        # 2. Portfolio Capacity Gate: If max positions already reached, do not wake LLM for new BUY setups
        if not has_open_position and portfolio_state:
            if portfolio_state.get("max_positions_reached", False):
                return False, "MAX_CONCURRENT_POSITIONS_REACHED"

        # 3. Check Cooldown Cache (avoid re-calling LLM every 40s after a HOLD unless price moved > 0.35%)
        now = time.time()
        last_price = features.get("last_price", 0.0)
        if symbol in self._hold_cooldown_cache:
            last_ts, cached_price = self._hold_cooldown_cache[symbol]
            price_change_pct = abs(last_price - cached_price) / (cached_price + 1e-9) * 100.0
            if (now - last_ts) < self.config.llm_hold_cooldown_sec and price_change_pct < 0.35:
                return False, "COOLDOWN_RECENT_LLM_HOLD"

        # 3. Microstructure & Volatility-to-Fee Gate
        spread = features.get("spread_pct", 1.0)
        if spread > self.config.max_spread_pct:
            return False, f"SPREAD_TOO_WIDE_{spread:.2f}%"

        atr_pct = features.get("atr_pct", 0.5)
        if atr_pct < self.config.min_atr_pct:
            return False, f"LOW_VOLATILITY_ATR_{atr_pct:.2f}%_BELOW_FEE_HURDLE"

        rsi = features.get("rsi_14", 50.0)
        obi = features.get("obi_top5", 0.0)
        taker_buy = features.get("taker_buy_ratio", 0.5)
        trend_bullish = features.get("trend_bullish", False)
        macro_uptrend = features.get("macro_uptrend", False)
        bullish_rebound = features.get("bullish_rebound", False)
        rsi_turning_up = features.get("rsi_turning_up", False)
        min_rsi_last3 = features.get("min_rsi_last3", 50.0)
        vol_z = features.get("volume_zscore", 0.0)
        roc_3 = features.get("roc_3_pct", 0.0)

        # Setup 1: Confirmed Macro Uptrend Pullback with Buyer Support
        if (
            macro_uptrend
            and trend_bullish
            and (38.0 <= rsi <= 62.0)
            and (obi >= 0.08 or taker_buy >= 0.52)
            and (bullish_rebound or rsi_turning_up or last_price >= features.get("ema9", 0))
        ):
            return True, "CONFIRMED_UPTREND_PULLBACK_REBOUND"

        # Setup 2: Order-Book Buyer Absorption in Confirmed Macro Uptrend (No Falling Knives!)
        if (
            macro_uptrend
            and (obi >= 0.25 and taker_buy >= 0.58)
            and (40.0 <= rsi <= 65.0)
            and (vol_z >= 0.0 or roc_3 >= 0.15)
        ):
            return True, "ORDERBOOK_BUYER_ABSORPTION"

        # Setup 3: High-Volume Momentum Breakout in Macro Uptrend
        if (
            macro_uptrend
            and (vol_z >= 1.2 or roc_3 >= 0.35)
            and (obi >= 0.10 or taker_buy >= 0.52)
            and (45.0 <= rsi <= 68.0)
        ):
            return True, "HIGH_VOLUME_MOMENTUM_BREAKOUT"

        return False, "NO_EDGE_MARKET_NEUTRAL"

    def evaluate_market(
        self,
        symbol: str,
        features: dict,
        portfolio_state: dict,
        recent_trades_summary: Optional[dict] = None,
    ) -> tuple[AIDecisionOutput, bool]:
        """
        Returns (AIDecisionOutput, was_llm_invoked: bool).
        """
        norm_symbol = MarketComplianceGuard.assert_market_allowed(symbol)
        has_pos = portfolio_state.get("position_qty", 0.0) > 0

        wake, trigger_reason = self.should_wake_llm(
            norm_symbol, features, has_open_position=has_pos, portfolio_state=portfolio_state
        )
        if not wake:
            return (
                AIDecisionOutput(
                    symbol=norm_symbol,
                    action="HOLD",
                    order_type="LIMIT",
                    confidence=0.5,
                    allocation_pct=0.0,
                    signals_observed=[trigger_reason],
                    reasoning_fa=f"فیلتر دروازه‌بان کمّی: عدم نیاز به فراخوانی مدل زبانی ({trigger_reason}).",
                ),
                False,
            )

        if not self.client:
            return self._quant_rule_fallback(norm_symbol, features, portfolio_state, trigger_reason), True

        user_payload = {
            "symbol": norm_symbol,
            "trigger": trigger_reason,
            "market": features,
            "recent_flow": recent_trades_summary or {},
            "portfolio": {
                "available_tmn": round(portfolio_state.get("available_tmn", 0), 0),
                "total_equity_tmn": round(portfolio_state.get("total_equity_tmn", 5_000_000), 0),
                "has_open_position": has_pos,
                "entry_price": portfolio_state.get("entry_price", None),
                "unrealized_pnl_pct": round(portfolio_state.get("unrealized_pnl_pct", 0.0), 4),
            },
        }

        messages = [
            {"role": "system", "content": COMPACT_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ]

        try:
            self.total_llm_calls += 1
            raw_content = self._call_llm_with_retry(messages)
            cleaned_json = self._clean_json_text(raw_content)
            parsed = json.loads(cleaned_json)
            parsed["symbol"] = norm_symbol
            decision = AIDecisionOutput(**parsed)

            if decision.action == "HOLD":
                self._hold_cooldown_cache[norm_symbol] = (time.time(), features.get("last_price", 0.0))
            return decision, True

        except Exception as e:
            logger.warning(f"LLM API Error on {norm_symbol} ({e}) -> Using deterministic quant fallback.")
            return self._quant_rule_fallback(norm_symbol, features, portfolio_state, trigger_reason), True

    def _call_llm_with_retry(self, messages: list) -> str:
        """Tries reasoning mode first (with ample token budget), then falls back to standard chat completion."""
        call_kwargs = {
            "model": self.model,
            "response_format": {"type": "json_object"},
            "messages": messages,
            "timeout": 20.0,
        }
        if self.reasoning_effort:
            try:
                resp = self.client.chat.completions.create(
                    **call_kwargs,
                    reasoning_effort=self.reasoning_effort,
                    max_completion_tokens=1500,  # Enough room for internal CoT + JSON output
                )
                content = resp.choices[0].message.content
                if content and "{" in content:
                    return content
            except Exception as inner_e:
                logger.debug(f"Reasoning parameter call failed, retrying standard completion: {inner_e}")

        resp = self.client.chat.completions.create(
            **call_kwargs,
            temperature=0.1,
            max_completion_tokens=600,
        )
        return resp.choices[0].message.content or "{}"

    def _quant_rule_fallback(
        self, symbol: str, features: dict, portfolio_state: dict, trigger_reason: str
    ) -> AIDecisionOutput:
        has_pos = portfolio_state.get("position_qty", 0.0) > 0
        last_price = features["last_price"]
        atr = features["atr_14"]
        best_bid = features["best_bid"]

        if has_pos:
            unrealized = portfolio_state.get("unrealized_pnl_pct", 0.0)
            if unrealized >= 1.2 and (features["rsi_14"] > 72.0 or features["obi_top5"] < -0.30):
                return AIDecisionOutput(
                    symbol=symbol,
                    action="SELL",
                    order_type="LIMIT",
                    confidence=0.85,
                    allocation_pct=0.0,
                    limit_price=best_bid,
                    signals_observed=["QUANT_PROFIT_TAKE_EXHAUSTION", f"PNL_{unrealized:+.2f}%"],
                    reasoning_fa=(
                        f"تحقق سود کمّی: موقعیت با بازدهی مثبت {unrealized:+.2f}% و نشانه اشباع خرید "
                        f"(RSI={features['rsi_14']:.1f}) در بهترین قیمت خرید ثبت شد."
                    ),
                )
            return AIDecisionOutput(
                symbol=symbol,
                action="HOLD",
                order_type="LIMIT",
                confidence=0.75,
                allocation_pct=0.0,
                signals_observed=["QUANT_POSITION_HOLD"],
                reasoning_fa="نگهداری پوزیشن باز؛ حد ضرر و تریلینگ استاپ فعال است و فضا برای رشد تا حد سود داده می‌شود.",
            )

        sl_price = max(last_price - 1.8 * atr, last_price * (1.0 - self.config.hard_stop_loss_pct))
        tp_price = max(last_price + 2.8 * atr, last_price * (1.0 + self.config.default_take_profit_pct))

        return AIDecisionOutput(
            symbol=symbol,
            action="BUY",
            order_type="LIMIT",
            confidence=0.76,
            allocation_pct=self.config.max_position_pct,
            limit_price=best_bid,
            stop_loss_price=round(sl_price, 2),
            take_profit_price=round(tp_price, 2),
            signals_observed=[
                trigger_reason,
                f"RSI={features['rsi_14']}",
                f"OBI={features['obi_top5']}",
                f"ATR%={features.get('atr_pct', 0)}%",
            ],
            reasoning_fa=(
                f"ورود الگوریتمی ({trigger_reason}): شاخص RSI={features['rsi_14']}، عدم تعادل دفتر سفارشات "
                f"OBI={features['obi_top5']}، نسبت خریداران Taker={features['taker_buy_ratio']} و نوسان ATR={features.get('atr_pct', 0)}٪. "
                f"حد ضرر در {sl_price:,.0f} تومان و حد سود در {tp_price:,.0f} تومان تنظیم شد."
            ),
        )