# dashboard.py
"""
Wallex Agentic AI Trading Command Center (Streamlit Dashboard).
Read-only live monitor and 5-minute demo video presentation tool.
Reads portfolio_state.json and decision_logs.jsonl with zero interference to main.py.
"""

import os
import json
import time
from datetime import datetime
import pandas as pd
import streamlit as st

from config import AgentConfig, ExecutionMode
from market_compliance import FORBIDDEN_SYMBOLS_NORMALIZED, PRIORITY_ALLOWED_TMN_MARKETS
from backtest_engine import StrategyBacktester

st.set_page_config(
    page_title="Wallex Agentic AI Trading Command Center",
    page_icon="📈",
    layout="wide",
)

# Custom CSS for Right-to-Left (RTL) Persian AI reasoning cards and KPI badges
st.markdown(
    """
    <style>
    .rtl-box {
        direction: rtl;
        text-align: right;
        font-family: 'Vazirmatn', 'Tahoma', sans-serif;
        background-color: #1e293b;
        color: #f8fafc;
        padding: 14px 18px;
        border-radius: 8px;
        border-right: 5px solid #38bdf8;
        margin-bottom: 10px;
        line-height: 1.7;
    }
    .risk-badge {
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def load_portfolio_state(state_file: str, initial_capital: float) -> dict:
    if os.path.exists(state_file):
        try:
            with open(state_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "shadow_available_tmn": initial_capital,
        "available_tmn": initial_capital,
        "total_equity_tmn": initial_capital,
        "peak_equity_tmn": initial_capital,
        "realized_pnl_tmn": 0.0,
        "total_trades_count": 0,
        "winning_trades_count": 0,
        "losing_trades_count": 0,
        "positions": {},
        "pending_buy_orders": {},
        "updated_at": "Not started",
    }


def load_decision_logs(log_file: str) -> list[dict]:
    records = []
    if os.path.exists(log_file):
        try:
            with open(log_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
        except Exception:
            pass
    return records


# ==========================================
# SIDEBAR: CONTROLS & CONTEST COMPLIANCE
# ==========================================
cfg = AgentConfig()

with st.sidebar:
    st.title("🤖 Wallex AI Agent")
    st.caption("StarCoach Agentic AI Hackathon")

    auto_refresh = st.toggle("Live Auto-Refresh (5s)", value=False)
    if st.button("🔄 Refresh Telemetry Now", width='stretch'):
        st.rerun()

    st.divider()
    st.subheader("🛡️ Active Risk Guardrails")
    st.markdown(
        f"""
        - **Initial Capital:** `{cfg.initial_capital_tmn:,.0f} TMN`
        - **Exchange Min Order:** `100,000 TMN`
        - **Safe Entry Floor:** `{cfg.min_order_tmn:,.0f} TMN`
        - **Max Position Size:** `{cfg.max_position_pct*100:.0f}%`
        - **Cash Reserve Floor:** `{cfg.min_cash_reserve_pct*100:.0f}%`
        - **Max Open Positions:** `{cfg.max_concurrent_positions}`
        - **Hard Stop-Loss Cap:** `-{cfg.hard_stop_loss_pct*100:.2f}%`
        - **Drawdown Breaker:** `-{cfg.max_portfolio_drawdown_pct*100:.1f}%`
        - **Forbidden Markets Blocked:** `{len(FORBIDDEN_SYMBOLS_NORMALIZED)} pairs`
        """
    )

state = load_portfolio_state(cfg.state_file, cfg.initial_capital_tmn)
logs = load_decision_logs(cfg.decision_log_file)

# Compute live portfolio KPIs
avail_cash = float(state.get("available_tmn", state.get("shadow_available_tmn", cfg.initial_capital_tmn)))
latest_equity = float(state.get("total_equity_tmn", avail_cash))
if logs and "portfolio_equity_tmn" in logs[-1]:
    latest_equity = float(logs[-1]["portfolio_equity_tmn"])

peak_equity = max(float(state.get("peak_equity_tmn", cfg.initial_capital_tmn)), latest_equity)
net_pnl_tmn = latest_equity - cfg.initial_capital_tmn
net_return_pct = (net_pnl_tmn / cfg.initial_capital_tmn) * 100.0
current_dd_pct = ((peak_equity - latest_equity) / max(peak_equity, 1.0)) * 100.0

total_trades = int(state.get("total_trades_count", 0))
win_trades = int(state.get("winning_trades_count", 0))
win_rate = (win_trades / total_trades * 100.0) if total_trades > 0 else 0.0
active_positions = state.get("positions", {})
pending_orders = state.get("pending_buy_orders", {})

# ==========================================
# TOP KPI METRIC CARDS (150 FINANCIAL PTS)
# ==========================================
st.markdown("## 📊 Wallex Autonomous Spot Trading Command Center")

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric(
    "Total Portfolio Equity",
    f"{latest_equity:,.0f} TMN",
    f"{net_pnl_tmn:+,.0f} TMN ({net_return_pct:+.2f}%)",
)
col2.metric(
    "Available Cash (TMN)",
    f"{avail_cash:,.0f} TMN",
    f"{(avail_cash / max(latest_equity, 1.0))*100:.1f}% liquid (Min 20%)",
)
col3.metric(
    "Current Drawdown",
    f"-{current_dd_pct:.2f}%",
    f"Breaker Limit: -3.50%",
    delta_color="inverse",
)
col4.metric(
    "Completed Trades & Win Rate",
    f"{total_trades} Trades",
    f"Win Rate: {win_rate:.1f}%",
)
col5.metric(
    "Open Positions / Pending",
    f"{len(active_positions)} / {cfg.max_concurrent_positions}",
    f"{len(pending_orders)} Resting Limits",
)

st.divider()

# ==========================================
# 4 PRESENTATION TABS FOR THE 5-MIN VIDEO
# ==========================================
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Live Equity & Open Positions",
    "🧠 Agentic AI Decision Logs (Persian)",
    "⏪ Backtest & Shadow Test Suite",
    "🛡️ Contest Compliance & Architecture",
])

# ---- TAB 1: EQUITY CURVE & ACTIVE POSITIONS ----
with tab1:
    c_left, c_right = st.columns([3, 2])

    with c_left:
        st.subheader("Portfolio Equity Curve (Realized + Mark-to-Market TMN)")
        if logs:
            eq_df = pd.DataFrame([
                {
                    "Time": r.get("timestamp", ""),
                    "Equity (TMN)": float(r.get("portfolio_equity_tmn", cfg.initial_capital_tmn)),
                    "Drawdown (%)": float(r.get("current_drawdown_pct", 0.0)),
                }
                for r in logs if r.get("portfolio_equity_tmn")
            ])
            if not eq_df.empty:
                st.line_chart(eq_df.set_index("Time")["Equity (TMN)"], height=320)
            else:
                st.info("Waiting for initial logged cycle equity points...")
        else:
            st.info("No decision logs recorded yet. Start `python main.py --mode SHADOW` or `LIVE`.")

    with c_right:
        st.subheader("Active Open Positions & Trailing Stops")
        if active_positions:
            pos_rows = []
            for sym, p in active_positions.items():
                pos_rows.append({
                    "Market": sym,
                    "Qty": p.get("quantity"),
                    "Entry (TMN)": f"{float(p.get('entry_price', 0)):,.0f}",
                    "Stop-Loss": f"{float(p.get('stop_loss_price', 0)):,.0f}",
                    "Take-Profit": f"{float(p.get('take_profit_price', 0)):,.0f}",
                    "Trailing Active": "✅ Yes" if p.get("trailing_active") else "⏳ Waiting",
                })
            st.dataframe(pd.DataFrame(pos_rows), width='stretch', hide_index=True)
        else:
            st.success("100% Realized in Tomans (TMN) — Zero Open Crypto Exposure.")

        st.subheader("Pending Limit Orders (3-Min Auto-Sweep)")
        if pending_orders:
            pend_rows = []
            for sym, po in pending_orders.items():
                age_s = int(time.time() - float(po.get("created_at_ts", time.time())))
                pend_rows.append({
                    "Symbol": sym,
                    "Limit Price": f"{float(po.get('limit_price', 0)):,.0f}",
                    "Allocated TMN": f"{float(po.get('order_tmn', 0)):,.0f}",
                    "Age (s)": f"{age_s}s / 180s",
                })
            st.dataframe(pd.DataFrame(pend_rows), width='stretch', hide_index=True)
        else:
            st.caption("No unfilled resting limit orders.")

# ---- TAB 2: AGENTIC AI DECISION LOGS (PERSIAN) ----
with tab2:
    st.subheader("🧠 Immutable AI Decision Audit Trail (`decision_logs.jsonl`)")
    f_col1, f_col2, f_col3 = st.columns(3)
    mode_filter = f_col1.selectbox("Filter by Execution Mode", ["ALL", "LIVE", "SHADOW"])
    action_filter = f_col2.selectbox("Filter by Action", ["ALL", "BUY", "SELL", "HOLD"])
    limit_count = f_col3.slider("Recent Decisions to Display", 5, 100, 25)

    filtered_logs = list(reversed(logs))
    if mode_filter != "ALL":
        filtered_logs = [r for r in filtered_logs if r.get("mode") == mode_filter]
    if action_filter != "ALL":
        filtered_logs = [r for r in filtered_logs if r.get("action") == action_filter]

    if not filtered_logs:
        st.warning("No matching entries in `decision_logs.jsonl` yet.")
    else:
        for entry in filtered_logs[:limit_count]:
            act = entry.get("action", "HOLD")
            icon = "🟢 BUY" if act == "BUY" else ("🔴 SELL" if act == "SELL" else "⚪ HOLD")
            sym = entry.get("symbol", "")
            ts = entry.get("timestamp", "")[:19].replace("T", " ")
            conf = float(entry.get("ai_confidence", 0.0)) * 100.0
            verdict = entry.get("risk_verdict", "")
            snap = entry.get("market_snapshot", {})
            trade_meta = entry.get("trade_metrics") or {}

            with st.expander(
                f"{icon} | {sym} | Mode: {entry.get('mode')} | Confidence: {conf:.0f}% | Verdict: {verdict} | {ts}",
                expanded=(act in ("BUY", "SELL")),
            ):
                # Render Persian AI reasoning in RTL box
                st.markdown(
                    f"<div class='rtl-box'><b>🧠 استدلال فنی هوش مصنوعی (AI Reasoning):</b><br>{entry.get('reasoning_fa', '')}</div>",
                    unsafe_allow_html=True,
                )
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Best Bid / Ask", f"{snap.get('best_bid', 0):,.0f}", f"Spread: {snap.get('spread_pct', 0):.3f}%")
                m2.metric("RSI(14) & Trend", f"{snap.get('rsi_14', 0)}", "Bullish (EMA9>21)" if snap.get("trend_bullish") else "Bearish")
                m3.metric("OrderBook Imbalance (OBI)", f"{snap.get('obi_top5', 0):+.3f}", f"Taker Buy: {snap.get('taker_buy_ratio', 0.5)*100:.1f}%")
                m4.metric("Volatility (ATR %)", f"{snap.get('atr_pct', 0):.2f}%", f"Vol Z-Score: {snap.get('volume_zscore', 0):+.2f}")

                if trade_meta:
                    st.json(trade_meta)

# ---- TAB 3: BACKTEST & SHADOW TEST RUNNER (20 PTS) ----
with tab3:
    st.subheader("⏪ Event-Driven Historical Backtester (Live Wallex Candles)")
    st.caption("Demonstrates strategy repeatability on historical Wallex OHLCV data with Wallex Credit Account fee modeling (0.25% Maker / 0.30% Taker).")

    b_col1, b_col2, b_col3 = st.columns([2, 2, 1])
    bt_symbol = b_col1.selectbox("Select Approved TMN Market", PRIORITY_ALLOWED_TMN_MARKETS, index=0)
    bt_hours = b_col2.slider("Historical Lookback (Hours)", 48, 336, 168, step=24)

    if b_col3.button("▶️ Run Backtest", type="primary", width='stretch'):
        with st.spinner(f"Fetching {bt_hours}h of 15m candles from Wallex and simulating trades on {bt_symbol}..."):
            bt_cfg = AgentConfig(mode=ExecutionMode.BACKTEST)
            tester = StrategyBacktester(bt_cfg, symbol=bt_symbol, lookback_hours=bt_hours)
            report = tester.run()
            if report:
                st.success(f"Backtest Completed on {bt_symbol} ({report['candles_analyzed']} candles)!")
                k1, k2, k3, k4, k5 = st.columns(5)
                k1.metric("Final Realized Capital", f"{report['final_capital_tmn']:,.0f} TMN", f"{report['return_pct']:+.2f}%")
                k2.metric("Net Realized PnL", f"{report['net_pnl_tmn']:+,.0f} TMN")
                k3.metric("Max Drawdown", f"-{report['max_drawdown_pct']:.2f}%", "Limit: 3.50%")
                k4.metric("Total Trades & Win Rate", f"{report['total_trades']} Trades", f"Win Rate: {report['win_rate_pct']:.1f}%")
                k5.metric("Profit Factor", f"{report['profit_factor']:.2f}")
            else:
                st.error("Could not fetch historical candles from Wallex API. Check network connection.")

# ---- TAB 4: CONTEST COMPLIANCE & ARCHITECTURE ----
with tab4:
    st.subheader("🛡️ Wallex Contest Compliance & Neuro-Symbolic Architecture")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            f"""
            ### 1. Strict Market Universe Enforcement
            - **Forbidden Pairs Blocked:** `{len(FORBIDDEN_SYMBOLS_NORMALIZED)}` symbols normalized and blocked at the execution client layer.
            - **Quote Currency Rule:** 100% locked to `*/TMN` spot pairs so all profits are realized directly in Tomans without double conversion fees.
            - **Approved High-Liquidity Whitelist:** `{', '.join(PRIORITY_ALLOWED_TMN_MARKETS)}`
            """
        )
    with c2:
        st.markdown(
            """
            ### 2. End-of-Contest Auto-Liquidation (`--flatten`)
            - **T-Minus 25 Minutes before Contest End:** Automatically cancels all open orders and sells 100% of remaining crypto balances (>100,000 TMN notional) into **Realized Tomans (TMN)**.
            - **Manual Override Command:** `python main.py --mode LIVE --flatten`
            """
        )

if auto_refresh:
    time.sleep(5)
    st.rerun()
