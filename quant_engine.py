# quant_engine.py
"""
Quantitative and Microstructure Feature Engine.
Computes technical indicators and order-book depth metrics for AI decision-making.
"""

from typing import Optional, List
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger("WallexAgent")


class QuantFeatureEngine:
    """
    Computes technical and microstructure indicators:
    - Trend: EMA(9) vs EMA(21), Trend Strength %
    - Momentum: RSI(14)
    - Volatility: ATR(14) for dynamic stop-loss positioning
    - Volume: Volume Z-score
    - Microstructure: Best Bid/Ask, Spread %, Top-5 Order Book Imbalance (OBI), Micro-price, Taker Buy Ratio
    """

    @staticmethod
    def compute_features(
        df: pd.DataFrame, orderbook: dict, recent_trades: Optional[List[dict]] = None
    ) -> dict:
        if df is None or len(df) < 25:
            return {"valid": False, "reason": "Insufficient candle data (<25 candles)"}

        bids = orderbook.get("bid", [])
        asks = orderbook.get("ask", [])
        if not bids or not asks:
            return {"valid": False, "reason": "Empty order book"}

        try:
            close = df["close"]
            high = df["high"]
            low = df["low"]
            vol = df["volume"]

            # 1. EMA 9 and 21
            ema9_series = close.ewm(span=9, adjust=False).mean()
            ema21_series = close.ewm(span=21, adjust=False).mean()
            ema9 = float(ema9_series.iloc[-1])
            ema21 = float(ema21_series.iloc[-1])
            trend_bullish = bool(ema9 > ema21)
            trend_strength_pct = round(((ema9 - ema21) / (ema21 + 1e-9)) * 100.0, 3)

            # 2. RSI 14
            delta = close.diff()
            gain = delta.clip(lower=0).rolling(14).mean()
            loss = (-delta.clip(upper=0)).rolling(14).mean()
            rs = gain / loss.replace(0, np.nan)
            rsi_val = 100 - (100 / (1 + rs.iloc[-1]))
            rsi = 50.0 if np.isnan(rsi_val) else float(rsi_val)

            # 3. ATR 14 (True Range)
            tr = pd.concat([
                high - low,
                (high - close.shift()).abs(),
                (low - close.shift()).abs()
            ], axis=1).max(axis=1)
            atr = float(tr.rolling(14).mean().iloc[-1])
            if np.isnan(atr) or atr <= 0:
                atr = float((high.iloc[-1] - low.iloc[-1]) or (close.iloc[-1] * 0.01))

            # 4. Volume Z-Score
            vol_mean = float(vol.rolling(20).mean().iloc[-1])
            vol_std = float(vol.rolling(20).std().iloc[-1])
            vol_z = float((vol.iloc[-1] - vol_mean) / (vol_std + 1e-9)) if vol_std > 0 else 0.0

            # 5. Order Book Microstructure (Top 5 levels)
            top_bids = bids[:5]
            top_asks = asks[:5]

            best_bid = float(top_bids[0]["price"])
            best_ask = float(top_asks[0]["price"])
            mid_price = (best_bid + best_ask) / 2.0
            spread_pct = ((best_ask - best_bid) / (mid_price + 1e-9)) * 100.0

            bid_vol_top5 = sum(float(b.get("quantity", 0.0)) for b in top_bids)
            ask_vol_top5 = sum(float(a.get("quantity", 0.0)) for a in top_asks)
            total_top5_vol = bid_vol_top5 + ask_vol_top5
            obi = (bid_vol_top5 - ask_vol_top5) / (total_top5_vol + 1e-9)

            # Micro-price (volume-weighted top of book)
            l1_bid_qty = float(top_bids[0].get("quantity", 1.0))
            l1_ask_qty = float(top_asks[0].get("quantity", 1.0))
            micro_price = (best_ask * l1_bid_qty + best_bid * l1_ask_qty) / (l1_bid_qty + l1_ask_qty + 1e-9)

            # Recent local support and resistance
            recent_low = float(low.tail(20).min())
            recent_high = float(high.tail(20).max())

            # Recent trades flow imbalance (Taker Buy Ratio)
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
                "last_price": float(close.iloc[-1]),
                "best_bid": best_bid,
                "best_ask": best_ask,
                "mid_price": round(mid_price, 2),
                "spread_pct": round(spread_pct, 4),
                "micro_price": round(micro_price, 2),
                "obi_top5": round(obi, 4),
                "taker_buy_ratio": taker_buy_ratio,
                "ema9": round(ema9, 2),
                "ema21": round(ema21, 2),
                "trend_bullish": trend_bullish,
                "trend_strength_pct": trend_strength_pct,
                "rsi_14": round(rsi, 2),
                "atr_14": round(atr, 2),
                "volume_zscore": round(vol_z, 2),
                "recent_low": round(recent_low, 2),
                "recent_high": round(recent_high, 2),
            }
        except Exception as e:
            logger.error(f"Error computing quant features: {e}")
            return {"valid": False, "reason": str(e)}
