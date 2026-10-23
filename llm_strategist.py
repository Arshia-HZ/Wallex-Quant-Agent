# llm_strategist.py
"""
Cost-Effective LLM Strategist with Quant Gatekeeper & Cooldown Cache.
Fixes reasoning_effort token starvation, strips markdown fences, filters out dead/flat markets,
and logs structured JSON trading decisions with Persian audit reasoning for hackathon judges.

Audit honesty: every decision carries `decision_source` ("LLM" | "QUANT_FALLBACK" | "QUANT_GATE")
and the LLM latency, so judges can see exactly which decisions the language model produced.
"""

import os
import re
import time
import json
import logging
from typing import Optional, Dict, Tuple
from openai import OpenAI
from models import AIDecisionOutput
from config import AgentConfig
from market_compliance import MarketComplianceGuard

logger = logging.getLogger("WallexAgent")

COMPACT_SYSTEM_PROMPT = """You are an institutional Algorithmic Spot Trading Agent in the Wallex Hackathon (5,000,000 TMN capital).
Your mission: Maximize realized TMN return until the contest deadline while keeping Maximum Drawdown strictly under 3.0%.
The quant gatekeeper has ALREADY found a candidate setup; you are the final independent judge and VETO layer.
Return HOLD unless you independently agree the trade has positive expectancy after ~0.55%-0.60% round-trip fees.
You must output ONLY valid JSON matching this schema:
{
  "symbol": "string",
  "action": "BUY" | "SELL" | "HOLD",
  "order_type": "LIMIT" | "MARKET",
  "confidence": float (0.0 to 1.0),
  "allocation_pct": float (0.0 to 0.40),
  "limit_price": float or null,
  "stop_loss_price": float or null,
  "take_profit_price": float or null,
  "signals_observed": ["string"],
  "reasoning_fa": "توضیح دقیق فنی به فارسی با ذکر اعداد واقعی RSI, EMA, OBI, ATR%, Spread, Stop-Loss برای داوران"
}

DECISION LOGIC & CONTEST RULES:
1. FEE & VOLATILITY GUARD: Wallex Credit Account charges 0.25% Maker and 0.30% Taker (~0.55%-0.60% round-trip). Never BUY if spread_pct > 0.55% or atr_pct < 0.35% (insufficient volatility to cover fees).
2. BUY requires confluence of independent signals:
   - Setup A (Trend Pullback): macro_uptrend is true, trend_bullish is true (EMA9 > EMA21), RSI_14 between 38 and 62, and (obi_top5 >= +0.08 or taker_buy_ratio >= 0.52).
   - Setup B (Orderbook Buyer Absorption): strong buyer depth (obi_top5 >= +0.25) and aggressive buying (taker_buy_ratio >= 0.58) with positive momentum.
   - Setup C (Momentum Breakout): macro_uptrend is true, volume_zscore >= 1.2 or roc_3_pct >= 0.35%, obi_top5 >= +0.10, and RSI_14 < 68.
   VETO (return HOLD) if any of these red flags is present: trend_bullish is false, price < ema50 (counter-trend), bb_position > 0.75 (overbought top), obi_top5 < 0 (negative buy depth), atr_pct < 0.35%, or bullish_rebound is false AND rsi_turning_up is false (falling knife).
3. Order Pricing: Entries are executed by the risk engine at the best ask (taker) because passive bids suffer adverse selection and save only 0.05% in fees. Set order_type "MARKET" and limit_price null for BUY.
4. Risk Bounds (enforced by the executor, which clamps anything outside these ranges):
   - stop_loss_price: between 0.8% and 1.4% below the entry price. Normally max(entry - 1.8 * atr_14, entry * (1 - 0.014)) floored at -0.8%. A tighter stop sits inside normal noise and loses stop distance PLUS fees. Use null to accept the engine default.
   - take_profit_price: at least +1.8% above entry (and at least 2x the stop distance); the executor ignores values outside +1.2%..+4.0%. Use null to accept the engine default.
5. Active Positions: DO NOT panic-sell at minor losses or intraday noise (0.55% fee friction will bleed equity). Stop-Loss, Trailing-Stop (+1.2% activation, breakeven lock) and a time stop are handled deterministically. Recommend SELL only if:
   - The position is IN NET PROFIT (unrealized_pnl_pct >= +1.2%) and showing overbought exhaustion (RSI_14 > 72 with weakening buy ratio) to lock in gains before a dip.
   - The position is stagnant (>2 hours) and macro trend broke down (price < EMA50).
   Otherwise, output action: "HOLD" to allow the trade breathing room to reach the Take-Profit target.
6. Always write a rich, specific Persian justification in 'reasoning_fa' citing exact indicator values.

Never ignore the risk limits above: a clean HOLD is always an acceptable answer."""


def compute_sl_tp(
    config: AgentConfig,
    price: float,
    atr: float,
    llm_sl: Optional[float] = None,
    llm_tp: Optional[float] = None,
) -> Tuple[float, float]:
    """
    Single source of truth for stop-loss / take-profit placement (used by the executor
    and by the quant fallback).

    Stop distance: 1.8*ATR, clamped to [min_stop_loss_pct, hard_stop_loss_pct].
    The LLM may choose a different stop only inside that same range.
    Take-profit distance: max(default_take_profit_pct, 2 x stop distance).
    The LLM may choose a different target only inside +1.2%..+4.0% and >= 1.5 x stop distance.
    """
    min_stop = config.min_stop_loss_pct
    max_stop = config.hard_stop_loss_pct
    atr_dist = (1.8 * atr / price) if (price > 0 and atr and atr > 0) else min_stop
    sl_dist = min(max(atr_dist, min_stop), max_stop)

    if llm_sl and 0 < llm_sl < price:
        d = (price - llm_sl) / price
        if min_stop <= d <= max_stop:
            sl_dist = d

    tp_dist = max(config.default_take_profit_pct, 2.0 * sl_dist)
    if llm_tp and llm_tp > price:
        d = (llm_tp - price) / price
        if 0.012 <= d <= 0.04 and d >= 1.5 * sl_dist:
            tp_dist = d

    return price * (1.0 - sl_dist), price * (1.0 + tp_dist)


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
        self.total_llm_failures = 0
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

        # 4. Microstructure & Volatility-to-Fee Gate
        spread = features.get("spread_pct", 1.0)
        if spread > self.config.max_spread_pct:
            return False, f"SPREAD_TOO_WIDE_{spread:.2f}%"

        atr_pct = features.get("atr_pct", 0.5)
        if atr_pct < max(0.35, self.config.min_atr_pct):
            return False, f"LOW_VOLATILITY_ATR_{atr_pct:.2f}%_BELOW_0.35%"

        # Sniper Mode Pre-Checks: Never wake LLM if fundamental entry vetoes are violated
        bb_pos = features.get("bb_position", 0.5)
        if bb_pos > 0.75:
            return False, f"OVERBOUGHT_BB_POSITION_{bb_pos:.2f}_ABOVE_0.75"

        trend_bullish = features.get("trend_bullish", False)
        if not trend_bullish:
            return False, "TREND_NOT_BULLISH"

        ema50 = features.get("ema50", 0.0)
        if ema50 > 0 and last_price < ema50:
            return False, f"PRICE_BELOW_EMA50_{last_price:,.0f}<{ema50:,.0f}"

        obi = features.get("obi_top5", 0.0)
        if obi < 0:
            return False, f"NEGATIVE_OBI_DEPTH_{obi:.2f}"

        rsi = features.get("rsi_14", 50.0)
        taker_buy = features.get("taker_buy_ratio", 0.5)
        macro_uptrend = features.get("macro_uptrend", False)
        bullish_rebound = features.get("bullish_rebound", False)
        rsi_turning_up = features.get("rsi_turning_up", False)
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
                    decision_source="QUANT_GATE",
                ),
                False,
            )

        if not self.client:
            fb = self._quant_rule_fallback(norm_symbol, features, portfolio_state, trigger_reason)
            return self._apply_sniper_vetoes_if_buy(norm_symbol, fb, features), True

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

        started = time.time()
        try:
            self.total_llm_calls += 1
            raw_content = self._call_llm_with_retry(messages)
            cleaned_json = self._clean_json_text(raw_content)
            parsed = json.loads(cleaned_json)
            parsed["symbol"] = norm_symbol
            # Never trust model-provided audit fields
            parsed.pop("decision_source", None)
            parsed.pop("llm_latency_ms", None)
            decision = AIDecisionOutput(**parsed)  # validators clamp confidence / allocation_pct
            decision.decision_source = "LLM"
            decision.llm_latency_ms = round((time.time() - started) * 1000.0, 0)
            decision = self._apply_sniper_vetoes_if_buy(norm_symbol, decision, features)

            if decision.action == "HOLD":
                self._hold_cooldown_cache[norm_symbol] = (time.time(), features.get("last_price", 0.0))
            return decision, True

        except Exception as e:
            self.total_llm_failures += 1
            logger.warning(f"LLM API Error on {norm_symbol} ({e}) -> Using deterministic quant fallback.")
            fb = self._quant_rule_fallback(norm_symbol, features, portfolio_state, trigger_reason)
            fb.llm_latency_ms = round((time.time() - started) * 1000.0, 0)
            fb = self._apply_sniper_vetoes_if_buy(norm_symbol, fb, features)
            return fb, True

    @staticmethod
    def check_sniper_vetoes(symbol: str, features: dict) -> Optional[str]:
        """
        Sniper Mode Entry Hard Vetoes:
        1. Trend Alignment Veto: Reject BUY if trend_bullish is False or price < ema50. Never buy counter-trend.
        2. Overbought Veto: Reject BUY if bb_position > 0.75 (prevent buying tops of Bollinger Bands).
        3. Microstructure Depth Veto: Reject BUY if obi_top5 < 0 (ensure buy depth actively exceeds sell depth).
        4. Minimum Volatility Filter: Ensure atr_pct >= 0.35% so bot does not enter dead ranges where fees exceed movement.
        """
        trend_bullish = bool(features.get("trend_bullish", False))
        price = float(features.get("last_price", 0.0) or features.get("best_ask", 0.0))
        ema50 = float(features.get("ema50", 0.0))
        bb_pos = float(features.get("bb_position", 0.5))
        obi = float(features.get("obi_top5", 0.0))
        atr_pct = float(features.get("atr_pct", 0.0))

        # 1. Trend Alignment Veto
        if not trend_bullish:
            return "SNIPER_VETO_COUNTER_TREND (trend_bullish is False)"
        if ema50 > 0 and price < ema50:
            return f"SNIPER_VETO_BELOW_EMA50 (price {price:,.0f} < EMA50 {ema50:,.0f})"

        # 2. Overbought Veto
        if bb_pos > 0.75:
            return f"SNIPER_VETO_OVERBOUGHT_BB (bb_position {bb_pos:.3f} > 0.75)"

        # 3. Microstructure Depth Veto
        if obi < 0:
            return f"SNIPER_VETO_NEGATIVE_OBI (obi_top5 {obi:+.4f} < 0)"

        # 4. Minimum Volatility Filter
        if atr_pct < 0.35:
            return f"SNIPER_VETO_LOW_VOLATILITY (atr_pct {atr_pct:.2f}% < 0.35%)"

        return None

    def _apply_sniper_vetoes_if_buy(
        self, symbol: str, decision: AIDecisionOutput, features: dict
    ) -> AIDecisionOutput:
        if decision.action != "BUY":
            return decision

        veto_reason = self.check_sniper_vetoes(symbol, features)
        if not veto_reason:
            return decision

        logger.info(
            f"[SNIPER VETO] {symbol} BUY converted to safe HOLD by sniper veto: {veto_reason}"
        )
        safe_hold = AIDecisionOutput(
            symbol=symbol,
            action="HOLD",
            order_type="LIMIT",
            confidence=0.5,
            allocation_pct=0.0,
            signals_observed=decision.signals_observed + [veto_reason],
            reasoning_fa=(
                f"فیلتر تک‌تیرانداز (Sniper Veto): سیگنال خرید به دلیل نقض شروط بنیادین "
                f"({veto_reason}) وتو شده و به نگهداری امن (HOLD) تبدیل گردید."
            ),
            decision_source=decision.decision_source,
            llm_latency_ms=decision.llm_latency_ms,
        )
        self._hold_cooldown_cache[symbol] = (time.time(), features.get("last_price", 0.0))
        return safe_hold

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
        """Deterministic decision used when the LLM is unavailable. Always labelled QUANT_FALLBACK."""
        has_pos = portfolio_state.get("position_qty", 0.0) > 0
        last_price = features["last_price"]
        atr = features["atr_14"]
        best_bid = features["best_bid"]
        best_ask = features["best_ask"]

        if has_pos:
            unrealized = portfolio_state.get("unrealized_pnl_pct", 0.0)
            if unrealized >= 1.2 and (features["rsi_14"] > 72.0 or features["obi_top5"] < -0.30):
                return AIDecisionOutput(
                    symbol=symbol,
                    action="SELL",
                    order_type="MARKET",
                    confidence=0.85,
                    allocation_pct=0.0,
                    limit_price=None,
                    signals_observed=["QUANT_PROFIT_TAKE_EXHAUSTION", f"PNL_{unrealized:+.2f}%"],
                    reasoning_fa=(
                        f"تحقق سود کمّی (جایگزین مدل زبانی): موقعیت با بازدهی مثبت {unrealized:+.2f}% و نشانه اشباع خرید "
                        f"(RSI={features['rsi_14']:.1f}) در بهترین قیمت خرید فروخته شد."
                    ),
                    decision_source="QUANT_FALLBACK",
                )
            return AIDecisionOutput(
                symbol=symbol,
                action="HOLD",
                order_type="LIMIT",
                confidence=0.75,
                allocation_pct=0.0,
                signals_observed=["QUANT_POSITION_HOLD"],
                reasoning_fa="نگهداری پوزیشن باز؛ حد ضرر و تریلینگ استاپ فعال است و فضا برای رشد تا حد سود داده می‌شود.",
                decision_source="QUANT_FALLBACK",
            )

        entry_ref = best_ask if self.config.force_market_entries else last_price
        sl_price, tp_price = compute_sl_tp(self.config, entry_ref, atr)

        return AIDecisionOutput(
            symbol=symbol,
            action="BUY",
            order_type="MARKET",
            confidence=0.76,
            allocation_pct=self.config.max_position_pct,
            limit_price=None,
            stop_loss_price=round(sl_price, 6),
            take_profit_price=round(tp_price, 6),
            signals_observed=[
                trigger_reason,
                f"RSI={features['rsi_14']}",
                f"OBI={features['obi_top5']}",
                f"ATR%={features.get('atr_pct', 0)}%",
                "LLM_UNAVAILABLE_QUANT_FALLBACK",
            ],
            reasoning_fa=(
                f"ورود الگوریتمی کمّی - مدل زبانی در دسترس نبود ({trigger_reason}): شاخص RSI={features['rsi_14']}، "
                f"عدم تعادل دفتر سفارشات OBI={features['obi_top5']}، نسبت خریداران Taker={features['taker_buy_ratio']} "
                f"و نوسان ATR={features.get('atr_pct', 0)}٪. "
                f"حد ضرر در {sl_price:,.0f} تومان و حد سود در {tp_price:,.0f} تومان تنظیم شد."
            ),
            decision_source="QUANT_FALLBACK",
        )