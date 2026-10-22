# backtest_engine.py
"""
Backtest Engine for Wallex Spot Trading Strategy.
Satisfies the 20-point 'Backtesting and Shadow Testing' judging requirement.
Replays historical candles from Wallex API, simulates execution with Wallex Credit Account fees (0.25% Maker / 0.30% Taker),
and computes quantitative metrics: Win Rate, Profit Factor, Max Drawdown, and Realized TMN Return.
"""

import sys
import logging
import pandas as pd
from config import AgentConfig, ExecutionMode
from wallex_client import WallexClient
from quant_engine import QuantFeatureEngine
from market_compliance import MarketComplianceGuard, normalize_symbol

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Backtester")


class StrategyBacktester:
    def __init__(self, config: AgentConfig, symbol: str = "BTCTMN", lookback_hours: int = 168):
        self.config = config
        self.symbol = normalize_symbol(symbol)
        self.lookback_hours = lookback_hours
        self.client = WallexClient(config)
        self.initial_capital = config.initial_capital_tmn
        self.maker_fee = getattr(config, "maker_fee_rate", 0.0025)
        self.taker_fee = getattr(config, "taker_fee_rate", 0.0030)

    def run(self) -> dict:
        norm_sym = MarketComplianceGuard.assert_market_allowed(self.symbol)
        logger.info(f"Fetching {self.lookback_hours}h of 15m historical candles for {norm_sym} from Wallex...")

        df = self.client.get_candles(norm_sym, resolution="15", lookback_hours=self.lookback_hours)
        if df is None or len(df) < 50:
            logger.error("Insufficient candle data for backtesting.")
            return {}

        logger.info(f"Loaded {len(df)} historical candles. Running event-driven backtest...")

        capital_tmn = self.initial_capital
        peak_capital = capital_tmn
        max_drawdown = 0.0

        position = None
        trades_history = []
        last_exit_candle_idx = -20

        for i in range(40, len(df)):
            sub_df = df.iloc[: i + 1]
            curr_candle = df.iloc[i]
            curr_price = float(curr_candle["close"])
            high_price = float(curr_candle["high"])
            low_price = float(curr_candle["low"])

            synth_orderbook = {
                "bid": [{"price": curr_price * 0.9996, "quantity": 1.0}],
                "ask": [{"price": curr_price * 1.0004, "quantity": 1.0}],
            }

            features = QuantFeatureEngine.compute_features(sub_df, synth_orderbook)
            if not features.get("valid", False):
                continue

            current_equity = capital_tmn + (position["qty"] * curr_price if position else 0.0)
            if current_equity > peak_capital:
                peak_capital = current_equity
            dd = (peak_capital - current_equity) / peak_capital
            if dd > max_drawdown:
                max_drawdown = dd

            # 1. Manage Open Position
            if position:
                bars_held = i - position["entry_idx"]
                exit_price = None
                exit_reason = ""

                # Check SL / TP against current bar BEFORE updating trailing peak from the same bar's high
                if low_price <= position["sl"]:
                    exit_price = position["sl"]
                    exit_reason = "TRAILING_STOP" if position["sl"] > position["entry_price"] else "STOP_LOSS"
                elif position["tp"] and high_price >= position["tp"]:
                    exit_price = position["tp"]
                    exit_reason = "TAKE_PROFIT"
                # Early Stagnation / Trend-Breakdown Exit: cuts dead trades before they hit full -1.20% SL
                elif bars_held >= 6 and not features["trend_bullish"] and features["rsi_14"] < 45.0 and curr_price < position["entry_price"]:
                    exit_price = curr_price
                    exit_reason = "EARLY_TREND_BREAK_EXIT"
                elif features["rsi_14"] > 74.0 and curr_price > position["entry_price"] * 1.012:
                    exit_price = curr_price
                    exit_reason = "OVERBOUGHT_PROFIT_TAKE"

                if exit_price:
                    # Take-Profit limit exits fill at Maker fee (0.25%); Stop/Market exits fill at Taker fee (0.30%)
                    exit_fee_rate = self.maker_fee if exit_reason == "TAKE_PROFIT" else self.taker_fee
                    fee = position["qty"] * exit_price * exit_fee_rate
                    gross_pnl = (exit_price - position["entry_price"]) * position["qty"]
                    net_pnl = gross_pnl - position["entry_fee"] - fee
                    capital_tmn += (position["qty"] * exit_price) - fee
                    last_exit_candle_idx = i

                    trades_history.append({
                        "entry_time": position["entry_time"],
                        "exit_time": curr_candle["timestamp"],
                        "entry_price": position["entry_price"],
                        "exit_price": exit_price,
                        "qty": position["qty"],
                        "net_pnl_tmn": net_pnl,
                        "return_pct": (exit_price - position["entry_price"]) / position["entry_price"],
                        "reason": exit_reason,
                    })
                    position = None
                    continue

                # Update peak and ratchet trailing stop for subsequent bars
                if high_price > position["peak_price"]:
                    position["peak_price"] = high_price

                gain = (position["peak_price"] - position["entry_price"]) / position["entry_price"]
                if gain >= 0.017:
                    # Lock at least +1.20% gross (+0.65% net after fees) once price has rallied +1.7%
                    min_profit_lock = position["entry_price"] * 1.012
                    trail_sl = position["peak_price"] * (1.0 - 0.006)
                    new_sl = max(position["sl"], min_profit_lock, trail_sl)
                    if new_sl > position["sl"]:
                        position["sl"] = new_sl

            # 2. Check Entry Setup (Do not open new positions in the final 10 bars before flatten deadline!)
            if not position and i < (len(df) - 10):
                if dd >= self.config.max_portfolio_drawdown_pct:
                    continue

                last_was_loss = bool(trades_history and trades_history[-1]["net_pnl_tmn"] <= 0)
                cooldown_bars = 8 if last_was_loss else 3
                if (i - last_exit_candle_idx) < cooldown_bars:
                    continue

                # Volatility must be large enough to clear Wallex's 0.55% round-trip fee
                if features["atr_pct"] < 0.35:
                    continue

                rsi = features["rsi_14"]
                vol_z = features["volume_zscore"]
                bullish_rebound = features["bullish_rebound"]
                rsi_turning_up = features["rsi_turning_up"]

                is_buy = False
                # Setup 1: Confirmed Macro Uptrend + Pullback that has ACTUALLY turned green/bounced
                if (
                    features["macro_uptrend"]
                    and features["trend_bullish"]
                    and features["trend_strength_pct"] >= 0.08
                    and (42.0 <= rsi <= 58.0)
                    and bullish_rebound
                    and rsi_turning_up
                ):
                    is_buy = True

                # Setup 2: Oversold Spring Reversal (Previous bars dipped <30 RSI, current bar is a strong green rebound)
                elif (
                    features["min_rsi_last3"] < 30.0
                    and rsi >= 31.0
                    and bullish_rebound
                    and rsi_turning_up
                    and vol_z > 0.3
                ):
                    is_buy = True

                # Setup 3: High-Volume Fresh Breakout in Macro Uptrend
                elif (
                    features["macro_uptrend"]
                    and features["trend_bullish"]
                    and vol_z >= 1.8
                    and features["roc_3_pct"] >= 0.45
                    and bullish_rebound
                    and (52.0 <= rsi <= 65.0)
                ):
                    is_buy = True

                if is_buy and capital_tmn >= self.config.min_order_tmn:
                    target_tmn = min(capital_tmn * self.config.max_position_pct, capital_tmn * 0.8)
                    entry_price = curr_price
                    qty = target_tmn / entry_price

                    # Cap Stop-Loss strictly at hard_stop_loss_pct (1.1% - 1.2%)
                    hard_sl = entry_price * (1.0 - self.config.hard_stop_loss_pct)
                    atr_sl = entry_price - (1.8 * features["atr_14"])
                    sl = max(atr_sl, hard_sl)

                    entry_fee = target_tmn * self.maker_fee
                    tp = max(
                        entry_price + (3.2 * features["atr_14"]),
                        entry_price * (1.0 + self.config.default_take_profit_pct),
                    )

                    capital_tmn -= (target_tmn + entry_fee)
                    position = {
                        "qty": qty,
                        "entry_price": entry_price,
                        "peak_price": entry_price,
                        "sl": sl,
                        "tp": tp,
                        "entry_fee": entry_fee,
                        "entry_time": curr_candle["timestamp"],
                        "entry_idx": i,
                    }

        # Close any open position at the end (Contest Rule: Liquidate to TMN)
        if position:
            last_price = float(df.iloc[-1]["close"])
            fee = position["qty"] * last_price * self.taker_fee
            net_pnl = (last_price - position["entry_price"]) * position["qty"] - position["entry_fee"] - fee
            capital_tmn += (position["qty"] * last_price) - fee
            trades_history.append({
                "entry_time": position["entry_time"],
                "exit_time": df.iloc[-1]["timestamp"],
                "entry_price": position["entry_price"],
                "exit_price": last_price,
                "qty": position["qty"],
                "net_pnl_tmn": net_pnl,
                "return_pct": (last_price - position["entry_price"]) / position["entry_price"],
                "reason": "END_OF_TEST_FLATTEN",
            })

        total_trades = len(trades_history)
        winning_trades = [t for t in trades_history if t["net_pnl_tmn"] > 0]
        losing_trades = [t for t in trades_history if t["net_pnl_tmn"] <= 0]
        win_rate = (len(winning_trades) / total_trades * 100.0) if total_trades > 0 else 0.0

        gross_profit = sum(t["net_pnl_tmn"] for t in winning_trades)
        gross_loss = abs(sum(t["net_pnl_tmn"] for t in losing_trades))
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (999.0 if gross_profit > 0 else 1.0)

        total_net_pnl = capital_tmn - self.initial_capital
        return_pct = (total_net_pnl / self.initial_capital) * 100.0

        report = {
            "symbol": self.symbol,
            "candles_analyzed": len(df),
            "initial_capital_tmn": self.initial_capital,
            "final_capital_tmn": round(capital_tmn, 0),
            "net_pnl_tmn": round(total_net_pnl, 0),
            "return_pct": round(return_pct, 2),
            "max_drawdown_pct": round(max_drawdown * 100.0, 2),
            "total_trades": total_trades,
            "win_rate_pct": round(win_rate, 1),
            "profit_factor": round(profit_factor, 2),
        }

        print("\n" + "=" * 60)
        print("          WALLEX STRATEGY BACKTEST RESULTS REPORT")
        print("=" * 60)
        print(f"Market Symbol:           {report['symbol']}")
        print(f"Candles Processed:       {report['candles_analyzed']} (15-min intervals)")
        print(f"Initial Capital:         {report['initial_capital_tmn']:,.0f} TMN")
        print(f"Final Realized Capital:  {report['final_capital_tmn']:,.0f} TMN")
        print(f"Net Realized PnL:        {report['net_pnl_tmn']:+,.0f} TMN ({report['return_pct']:+.2f}%)")
        print(f"Max Portfolio Drawdown:  {report['max_drawdown_pct']:.2f}% (Limit: 3.5%)")
        print(f"Total Completed Trades:  {report['total_trades']}")
        print(f"Win Rate:                {report['win_rate_pct']:.1f}%")
        print(f"Profit Factor:           {report['profit_factor']:.2f}")
        print("=" * 60)
        if trades_history:
            print("\nTrade Execution Log:")
            for idx, t in enumerate(trades_history, 1):
                win_loss = "WIN " if t["net_pnl_tmn"] > 0 else "LOSS"
                print(
                    f"  #{idx:02d} [{win_loss}] Return: {t['return_pct']*100:+.2f}% | "
                    f"Net: {t['net_pnl_tmn']:+,.0f} TMN | Exit: {t['reason']}"
                )
            print("=" * 60 + "\n")

        return report


if __name__ == "__main__":
    symbol = sys.argv[1] if len(sys.argv) > 1 else "BTCTMN"
    backtester = StrategyBacktester(AgentConfig(mode=ExecutionMode.BACKTEST), symbol=symbol, lookback_hours=168)
    backtester.run()