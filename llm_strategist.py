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
1. FEE & VOLATILITY GUARD: Wallex Credit Account charges 0.25% Maker and 0.30% Taker (~0.55%-0.60% round-trip). Never BUY if spread_pct > 0.20% or atr_pct < 0.40% (insufficient volatility to cover fees).
2. BUY requires confluence of at least 2 independent signals:
   - Setup A (Trend Pullback): trend_bullish is true (EMA9 > EMA21), RSI_14 between 36 and 63, and (obi_top5 >= +0.10 or taker_buy_ratio >= 0.55).
   - Setup B (Mean-Reversion Bounce): RSI_14 < 34 or bb_position < 0.15, combined with strong buyer absorption (obi_top5 >= +0.18 or taker_buy_ratio >= 0.56).
   - Setup C (Momentum Breakout): volume_zscore >= 1.5, roc_3_pct > 0.35%, obi_top5 >= +0.15, and RSI_14 < 67.
3. Order Pricing: Prefer LIMIT order at best_bid (for BUY) to capture Maker fee (0.25%), or MARKET only if volume_zscore > 2.0 and obi_top5 > 0.30.
4. Risk Bounds: Set stop_loss_price = min(last_price - 4.0 * atr_14, last_price * 0.988) — use min() to pick the WIDER stop. Set take_profit_price = max(last_price + 3.0 * (last_price - stop_loss_price), last_price * 1.024).
5. Active Positions: Recommend SELL if trend reverses strongly (EMA9 < EMA21 and obi_top5 < -0.15) or RSI_14 > 73 with weakening taker_buy_ratio < 0.45.
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

    def should_wake_llm(self, symbol: str, features: dict, has_open_position: bool) -> tuple[bool, str]:
        if not features.get("valid", False):
            return False, features.get("reason", "INVALID_DATA")

        # 1. If we hold an open position, wake LLM only if meaningful PnL move or reversal signal occurs
        if has_open_position:
            if (not features.get("trend_bullish", True) and features.get("obi_top5", 0.0) < -0.10) or features.get("rsi_14", 50.0) > 70.0:
                return True, "ACTIVE_POSITION_REVERSAL_CHECK"
            return False, "POSITION_MONITORED_BY_DETERMINISTIC_SL_TP"

        # 2. Check Cooldown Cache (avoid re-calling LLM every 40s after a HOLD unless price moved > 0.35%)
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
        vol_z = features.get("volume_zscore", 0.0)
        bb_pos = features.get("bb_position", 0.5)
        roc_3 = features.get("roc_3_pct", 0.0)

        # Setup 1: Bullish trend pullback with order book or trade flow support
        if trend_bullish and (36.0 <= rsi <= 63.0) and (obi >= 0.10 or taker_buy >= 0.56):
            return True, "TREND_PULLBACK_WITH_FLOW_SUPPORT"

        # Setup 2: Oversold / Lower Bollinger Band bounce with buyer absorption
        if (rsi < 34.0 or bb_pos < 0.15) and (obi >= 0.16 or taker_buy >= 0.55):
            return True, "OVERSOLD_BOLLINGER_REVERSAL_SETUP"

        # Setup 3: High-volume momentum breakout
        if (vol_z >= 1.5 or roc_3 >= 0.40) and obi >= 0.14 and rsi < 67.0:
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

        wake, trigger_reason = self.should_wake_llm(norm_symbol, features, has_open_position=has_pos)
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
            max_tokens=600,
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
            if not features["trend_bullish"] and features["obi_top5"] < -0.15 and unrealized > 0.3:
                return AIDecisionOutput(
                    symbol=symbol,
                    action="SELL",
                    order_type="MARKET",
                    confidence=0.82,
                    allocation_pct=0.0,
                    signals_observed=["QUANT_REVERSAL_EXIT", f"OBI={features['obi_top5']}"],
                    reasoning_fa=(
                        f"خروج تاکتیکی به علت چرخش روند (EMA9 زیر EMA21) و فشار فروش در اردربوک "
                        f"(OBI={features['obi_top5']}) جهت حفظ سود ({unrealized:.2f}%)."
                    ),
                )
            return AIDecisionOutput(
                symbol=symbol,
                action="HOLD",
                order_type="LIMIT",
                confidence=0.65,
                allocation_pct=0.0,
                signals_observed=["QUANT_POSITION_HOLD"],
                reasoning_fa="نگهداری پوزیشن باز؛ حد ضرر و تریلینگ استاپ فعال است.",
            )

        sl_price = max(last_price - 1.5 * atr, last_price * (1.0 - self.config.hard_stop_loss_pct))
        tp_price = max(last_price + 2.4 * atr, last_price * (1.0 + self.config.default_take_profit_pct))

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