# quant_engine.py
"""
Quantitative and Microstructure Feature Engine.
Includes Macro EMA50 slope, Bullish Candle Reversal confirmation, Wilder's RSI slope,
ATR fee-hurdle ratio, and Order Book Microstructure.
"""

from typing import Optional, List
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger("WallexAgent")


class QuantFeatureEngine:
    @staticmethod
    def compute_features(
        df: pd.DataFrame, orderbook: dict, recent_trades: Optional[List[dict]] = None
    ) -> dict:
        if df is None or df.empty or len(df) < 35:
            return {"valid": False, "reason": "Insufficient candle data (<35 candles)"}

        bids = orderbook.get("bid", [])
        asks = orderbook.get("ask", [])
        if not bids or not asks:
            return {"valid": False, "reason": "Empty order book"}

        try:
            open_s = df["open"].astype(float)
            close = df["close"].astype(float)
            high = df["high"].astype(float)
            low = df["low"].astype(float)
            vol = df["volume"].astype(float)

            last_price = float(close.iloc[-1])
            prev_close = float(close.iloc[-2])
            curr_open = float(open_s.iloc[-1])

            # 1. Multi-EMA Regime (EMA9, EMA21, EMA50) & Slope Confirmation
            ema9_series = close.ewm(span=9, adjust=False).mean()
            ema21_series = close.ewm(span=21, adjust=False).mean()
            ema50_series = close.ewm(span=50, adjust=False).mean()

            ema9 = float(ema9_series.iloc[-1])
            ema21 = float(ema21_series.iloc[-1])
            ema50 = float(ema50_series.iloc[-1])
            ema21_rising = bool(ema21 > float(ema21_series.iloc[-3]))
            ema50_rising = bool(ema50 >= float(ema50_series.iloc[-3]))

            trend_bullish = bool(ema9 > ema21 and ema21_rising)
            macro_uptrend = bool(last_price > ema50 and ema21 >= ema50 * 0.998 and ema50_rising)
            trend_strength_pct = round(((ema9 - ema21) / (ema21 + 1e-9)) * 100.0, 3)

            # Candle action confirmation (Prevents buying into red falling knives!)
            candle_range = max(float(high.iloc[-1] - low.iloc[-1]), 1e-9)
            close_in_upper_half = bool((last_price - float(low.iloc[-1])) / candle_range >= 0.55)
            is_green_candle = bool(last_price > curr_open and last_price > prev_close)
            bullish_rebound = bool(is_green_candle and close_in_upper_half)

            # Short-term 3-candle momentum (%)
            roc_3 = round(((last_price - float(close.iloc[-4])) / (float(close.iloc[-4]) + 1e-9)) * 100.0, 3)

            # 2. Wilder's RSI(14) + Previous RSI to confirm upward turn
            delta = close.diff()
            gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
            loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
            rs = gain / loss.replace(0, np.nan)
            rsi_series = 100.0 - (100.0 / (1.0 + rs))
            rsi = 50.0 if np.isnan(rsi_series.iloc[-1]) else float(rsi_series.iloc[-1])
            prev_rsi = 50.0 if np.isnan(rsi_series.iloc[-2]) else float(rsi_series.iloc[-2])
            min_rsi_last3 = float(rsi_series.tail(4).min()) if not rsi_series.tail(4).isna().all() else rsi
            rsi_turning_up = bool(rsi > prev_rsi + 0.8)

            # 3. ATR(14) & ATR % of Price
            tr = pd.concat([
                high - low,
                (high - close.shift()).abs(),
                (low - close.shift()).abs(),
            ], axis=1).max(axis=1)
            atr = float(tr.rolling(14).mean().iloc[-1])
            if np.isnan(atr) or atr <= 0:
                atr = float((high.iloc[-1] - low.iloc[-1]) or (last_price * 0.008))
            atr_pct = round((atr / (last_price + 1e-9)) * 100.0, 4)

            # 4. Bollinger Bands (20, 2.0)
            bb_mid = float(close.rolling(20).mean().iloc[-1])
            bb_std = float(close.rolling(20).std().iloc[-1])
            bb_upper = bb_mid + 2.0 * bb_std
            bb_lower = bb_mid - 2.0 * bb_std
            bb_pos = (last_price - bb_lower) / ((bb_upper - bb_lower) + 1e-9)

            # 5. Volume Z-Score
            vol_mean = float(vol.rolling(20).mean().iloc[-1])
            vol_std = float(vol.rolling(20).std().iloc[-1])
            vol_z = float((vol.iloc[-1] - vol_mean) / (vol_std + 1e-9)) if vol_std > 0 else 0.0

            # 6. Order Book Microstructure (Top 5 levels)
            top_bids = bids[:5]
            top_asks = asks[:5]

            best_bid = float(top_bids[0]["price"])
            best_ask = float(top_asks[0]["price"])
            mid_price = (best_bid + best_ask) / 2.0
            spread_pct = ((best_ask - best_bid) / (mid_price + 1e-9)) * 100.0

            bid_vol_top5 = sum(float(b.get("quantity", 0.0)) for b in top_bids)
            ask_vol_top5 = sum(float(a.get("quantity", 0.0)) for a in top_asks)
            obi = (bid_vol_top5 - ask_vol_top5) / (bid_vol_top5 + ask_vol_top5 + 1e-9)

            l1_bid_qty = float(top_bids[0].get("quantity", 1.0))
            l1_ask_qty = float(top_asks[0].get("quantity", 1.0))
            micro_price = (best_ask * l1_bid_qty + best_bid * l1_ask_qty) / (l1_bid_qty + l1_ask_qty + 1e-9)

            recent_low = float(low.tail(20).min())
            recent_high = float(high.tail(20).max())

            # 7. Recent Trade Flow Imbalance (Taker Buy Ratio)
            taker_buy_ratio = 0.5
            if recent_trades:
                buy_vol = 0.0
                total_vol = 0.0
                for t in recent_trades[:30]:
                    q = float(t.get("quantity", 0.0))
                    total_vol += q
                    if t.get("isBuyOrder", False):
                        buy_vol += q
                if total_vol > 0:
                    taker_buy_ratio = round(buy_vol / total_vol, 3)

            return {
                "valid": True,
                "last_price": last_price,
                "best_bid": best_bid,
                "best_ask": best_ask,
                "mid_price": round(mid_price, 2),
                "spread_pct": round(spread_pct, 4),
                "micro_price": round(micro_price, 2),
                "obi_top5": round(obi, 4),
                "taker_buy_ratio": taker_buy_ratio,
                "ema9": round(ema9, 2),
                "ema21": round(ema21, 2),
                "ema50": round(ema50, 2),
                "trend_bullish": trend_bullish,
                "macro_uptrend": macro_uptrend,
                "trend_strength_pct": trend_strength_pct,
                "bullish_rebound": bullish_rebound,
                "rsi_turning_up": rsi_turning_up,
                "min_rsi_last3": round(min_rsi_last3, 2),
                "roc_3_pct": roc_3,
                "rsi_14": round(rsi, 2),
                "atr_14": float(atr),
                "atr_pct": atr_pct,
                "bb_position": round(float(bb_pos), 3),
                "volume_zscore": round(vol_z, 2),
                "recent_low": float(recent_low),
                "recent_high": float(recent_high),
            }
        except Exception as e:
            logger.error(f"Error computing quant features: {e}")
            return {"valid": False, "reason": str(e)}