# dashboard.py
"""
Wallex Agentic AI Trading Command Center (Streamlit Dashboard).
Real-time monitoring console, interactive analytics, and risk telemetry.
Reads portfolio state and decision logs concurrently with zero interference to main.py.
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
    page_title="Wallex AI Trading Command Center",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling: Modern dark quant styling with RTL Persian cards and badge highlights
st.markdown(
    """
    <style>
    .main-header {
        font-size: 1.85rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        margin-bottom: 0.25rem;
    }
    .sub-caption {
        color: #94a3b8;
        font-size: 0.95rem;
        margin-bottom: 1.2rem;
    }
    .rtl-box {
        direction: rtl;
        text-align: right;
        font-family: 'Vazirmatn', 'Tahoma', system-ui, -apple-system, sans-serif;
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        color: #f8fafc;
        padding: 16px 20px;
        border-radius: 10px;
        border-right: 5px solid #0284c7;
        margin: 12px 0;
        line-height: 1.8;
        font-size: 0.98rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    }
    .chip-badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 6px;
        margin-bottom: 4px;
    }
    .chip-buy {
        background-color: rgba(16, 185, 129, 0.2);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.4);
    }
    .chip-sell {
        background-color: rgba(239, 68, 68, 0.2);
        color: #f87171;
        border: 1px solid rgba(239, 68, 68, 0.4);
    }
    .chip-hold {
        background-color: rgba(100, 116, 139, 0.2);
        color: #cbd5e1;
        border: 1px solid rgba(100, 116, 139, 0.4);
    }
    .metric-card-container {
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 12px 16px;
        background-color: #0f172a;
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
        "mode": "UNKNOWN",
        "shadow_available_tmn": initial_capital,
        "available_tmn": initial_capital,
        "total_equity_tmn": initial_capital,
        "peak_equity_tmn": initial_capital,
        "baseline_equity_tmn": initial_capital,
        "baseline_set": False,
        "realized_pnl_tmn": 0.0,
        "total_trades_count": 0,
        "winning_trades_count": 0,
        "losing_trades_count": 0,
        "positions": {},
        "pending_buy_orders": {},
        "pending_sell_orders": {},
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
                        try:
                            records.append(json.loads(line))
                        except Exception:
                            continue
        except Exception:
            pass
    return records


# ==========================================
# CONFIG & SIDEBAR CONTROLS
# ==========================================
cfg = AgentConfig()

with st.sidebar:
    st.markdown("### 🤖 Wallex AI Agent")
    st.caption("StarCoach Agentic AI Hackathon • Spot Trading")

    # Environment Selector (Live vs Shadow)
    mode_options = ["LIVE (Production)", "SHADOW (Simulation)"]
    has_live_log = os.path.exists("decision_logs_live.jsonl")
    default_idx = 0 if (cfg.mode == ExecutionMode.LIVE or has_live_log) else 1

    selected_env = st.radio(
        "Trading Environment",
        mode_options,
        index=default_idx,
        help="Switch between live production logs and shadow paper-trading simulation logs.",
    )
    is_live = "LIVE" in selected_env

    state_file = "portfolio_state_live.json" if is_live else "portfolio_state.json"
    log_file = "decision_logs_live.jsonl" if is_live else "shadow_decision_logs.jsonl"

    st.divider()

    # Telemetry Refresh Controls
    c_ref1, c_ref2 = st.columns([1, 1])
    auto_refresh = c_ref1.toggle("Auto-Refresh", value=False)
    refresh_rate = c_ref2.selectbox("Interval", [5, 10, 30], index=0, format_func=lambda s: f"{s}s")
    if st.button("🔄 Refresh Data Now", width="stretch"):
        st.rerun()

    st.divider()

    st.markdown("#### 🛡️ Active Risk Guardrails")
    st.markdown(
        f"""
        - **Initial Capital:** `{cfg.initial_capital_tmn:,.0f} TMN`
        - **Exchange Min Order:** `100,000 TMN`
        - **Safe Order Floor:** `{cfg.min_order_tmn:,.0f} TMN`
        - **Kelly High-Conviction:** `{getattr(cfg, 'high_conviction_position_pct', 0.35)*100:.0f}%`
        - **Standard Max Size:** `{cfg.max_position_pct*100:.0f}%`
        - **Cash Reserve Floor:** `{cfg.min_cash_reserve_pct*100:.0f}%`
        - **Max Concurrent Pos:** `{cfg.max_concurrent_positions}`
        - **Hard Stop-Loss Cap:** `-{cfg.hard_stop_loss_pct*100:.2f}%`
        - **Drawdown Breaker:** `-{cfg.max_portfolio_drawdown_pct*100:.1f}%`
        - **Active Universe:** `{len(PRIORITY_ALLOWED_TMN_MARKETS)} TMN Pairs`
        """
    )
    st.caption(f"State: `{state_file}`\nLogs: `{log_file}`")


# ==========================================
# LOAD TELEMETRY DATA
# ==========================================
state = load_portfolio_state(state_file, cfg.initial_capital_tmn)
logs = load_decision_logs(log_file)

# Extract completed trades from decision logs
closed_trades = [r for r in logs if r.get("action") == "SELL" and r.get("trade_metrics")]

# Baseline capital: use real recorded baseline or fallback to initial capital
baseline_equity = float(state.get("baseline_equity_tmn") or cfg.initial_capital_tmn)
avail_cash = float(state.get("available_tmn", state.get("shadow_available_tmn", cfg.initial_capital_tmn)))

# Latest equity from decision log cycle points
equity_logs = [r for r in logs if r.get("portfolio_equity_tmn") is not None]
if equity_logs:
    latest_equity = float(equity_logs[-1]["portfolio_equity_tmn"])
    current_dd_pct = float(equity_logs[-1].get("current_drawdown_pct", 0.0))
else:
    latest_equity = float(state.get("total_equity_tmn", avail_cash))
    peak_eq = max(float(state.get("peak_equity_tmn", baseline_equity)), latest_equity)
    current_dd_pct = ((peak_eq - latest_equity) / max(peak_eq, 1.0)) * 100.0

# Net performance relative to verified baseline
net_pnl_tmn = latest_equity - baseline_equity
net_return_pct = (net_pnl_tmn / max(baseline_equity, 1.0)) * 100.0

# Trade counts & win rate
if closed_trades:
    total_trades = len(closed_trades)
    win_trades = sum(1 for t in closed_trades if float(t["trade_metrics"].get("realized_net_pnl_tmn", 0)) > 0)
    loss_trades = total_trades - win_trades
    win_rate = (win_trades / total_trades * 100.0) if total_trades > 0 else 0.0
    total_realized_pnl = sum(float(t["trade_metrics"].get("realized_net_pnl_tmn", 0)) for t in closed_trades)
else:
    total_trades = int(state.get("total_trades_count", 0))
    win_trades = int(state.get("winning_trades_count", 0))
    win_rate = (win_trades / total_trades * 100.0) if total_trades > 0 else 0.0
    total_realized_pnl = float(state.get("realized_pnl_tmn", 0.0))

active_positions = state.get("positions", {})
pending_buys = state.get("pending_buy_orders", {})
pending_sells = state.get("pending_sell_orders", {})
total_pending = len(pending_buys) + len(pending_sells)


# ==========================================
# HEADER & TOP-LEVEL FINANCIAL KPIS
# ==========================================
env_badge = "🟢 LIVE PRODUCTION" if is_live else "🟡 SHADOW SIMULATION"
st.markdown(f"<div class='main-header'>Wallex AI Spot Trading Command Center</div>", unsafe_allow_html=True)
st.markdown(f"<div class='sub-caption'>Mode: <b>{env_badge}</b> • Active Universe: <b>14 High-Volume TMN Pairs</b> • Baseline Equity: <b>{baseline_equity:,.0f} TMN</b></div>", unsafe_allow_html=True)

kpi_col1, kpi_col2, kpi_col3, kpi_col4, kpi_col5 = st.columns(5)

kpi_col1.metric(
    "Total Portfolio Equity",
    f"{latest_equity:,.0f} TMN",
    f"{net_pnl_tmn:+,.0f} TMN ({net_return_pct:+.2f}%)",
    delta_color="normal",
)

liquid_ratio = (avail_cash / max(latest_equity, 1.0)) * 100.0
kpi_col2.metric(
    "Available Liquid Cash",
    f"{avail_cash:,.0f} TMN",
    f"{liquid_ratio:.1f}% Liquid (Floor 20%)",
    delta_color="off",
)

kpi_col3.metric(
    "Current Drawdown",
    f"-{current_dd_pct:.2f}%",
    f"Max Breaker: -{cfg.max_portfolio_drawdown_pct*100:.1f}%",
    delta_color="inverse",
)

kpi_col4.metric(
    "Completed Trades",
    f"{total_trades} Trades",
    f"Win Rate: {win_rate:.1f}% ({win_trades}W / {total_trades - win_trades}L)",
    delta_color="off",
)

kpi_col5.metric(
    "Positions & Orders",
    f"{len(active_positions)} / {cfg.max_concurrent_positions} Active",
    f"{total_pending} In-Flight Orders",
    delta_color="off",
)

st.divider()


# ==========================================
# TABBED WORKSPACE
# ==========================================
tab_perf, tab_pos, tab_ai, tab_backtest, tab_rules = st.tabs([
    "📈 Performance & Equity Curve",
    "🎯 Active Positions & Orders",
    "🧠 AI Strategist Decisions",
    "⚡ Event-Driven Backtest",
    "🛡️ Risk Rules & Architecture",
])


# ------------------------------------------------------------------
# TAB 1: PERFORMANCE & EQUITY CURVE
# ------------------------------------------------------------------
with tab_perf:
    p_left, p_right = st.columns([3, 2])

    with p_left:
        st.subheader("Portfolio Equity Trajectory (Mark-to-Market TMN)")
        if equity_logs:
            chart_records = []
            for r in equity_logs:
                chart_records.append({
                    "Time": r.get("timestamp", "")[:19].replace("T", " "),
                    "Equity (TMN)": float(r.get("portfolio_equity_tmn", baseline_equity)),
                    "Drawdown (%)": float(r.get("current_drawdown_pct", 0.0)),
                })
            eq_df = pd.DataFrame(chart_records).drop_duplicates(subset=["Time"])
            if not eq_df.empty:
                st.line_chart(eq_df.set_index("Time")["Equity (TMN)"], height=310)
                st.caption("Equity incorporates realized cash plus mark-to-market valuations of active crypto positions.")
            else:
                st.info("Awaiting initial equity log datapoints...")
        else:
            st.info(f"No equity cycle logs found in `{log_file}` yet. Ensure trading runner is active.")

    with p_right:
        st.subheader("Drawdown Profile (%)")
        if equity_logs and not eq_df.empty:
            st.area_chart(eq_df.set_index("Time")["Drawdown (%)"], height=310, color="#ef4444")
            st.caption(f"Peak-to-trough drawdown from highest observed equity (Circuit breaker: -{cfg.max_portfolio_drawdown_pct*100:.1f}%).")
        else:
            st.info("Awaiting drawdown trajectory data...")

    st.divider()

    # Closed Trades History Table
    st.subheader("📋 Closed Trades Audit Trail")
    if closed_trades:
        trade_rows = []
        for t in reversed(closed_trades):
            meta = t.get("trade_metrics", {})
            exec_res = t.get("execution_result", {})
            entry_px = float(meta.get("entry_price", 0))
            exit_px = float(meta.get("exit_price", 0))
            sold_qty = float(meta.get("sold_quantity", 0))
            pnl_tmn = float(meta.get("realized_net_pnl_tmn", 0))
            pnl_pct = float(meta.get("realized_net_pnl_pct", 0))
            verdict = t.get("risk_verdict", "EXIT")
            ts = t.get("timestamp", "")[:19].replace("T", " ")

            trade_rows.append({
                "Timestamp (UTC)": ts,
                "Symbol": t.get("symbol", ""),
                "Exit Reason": verdict.replace("_", " "),
                "Entry Price": f"{entry_px:,.0f} TMN",
                "Exit Price": f"{exit_px:,.0f} TMN",
                "Sold Qty": f"{sold_qty:g}",
                "Net PnL (TMN)": f"{pnl_tmn:+,.0f} TMN",
                "Net Return (%)": f"{pnl_pct:+.2f}%",
                "Status": "🟢 WIN" if pnl_tmn > 0 else "🔴 LOSS",
            })

        st.dataframe(pd.DataFrame(trade_rows), width="stretch", hide_index=True)
    else:
        st.info("No completed closed trades in the selected environment yet.")


# ------------------------------------------------------------------
# TAB 2: ACTIVE POSITIONS & IN-FLIGHT ORDERS
# ------------------------------------------------------------------
with tab_pos:
    pos_col, ord_col = st.columns([3, 2])

    with pos_col:
        st.subheader("Active Crypto Positions")
        if active_positions:
            pos_table = []
            for sym, p in active_positions.items():
                entry_px = float(p.get("entry_price", 0))
                sl_px = float(p.get("stop_loss_price", 0))
                tp_px = float(p.get("take_profit_price", 0))
                qty = float(p.get("quantity", 0))
                cost = float(p.get("cost_tmn", 0) or (qty * entry_px * 1.003))
                peak_px = float(p.get("highest_price", entry_px))
                trailing_status = "🟢 ACTIVE" if p.get("trailing_active") else "⏳ ARMED (at +1.0%)"

                pos_table.append({
                    "Symbol": sym,
                    "Quantity": f"{qty:g}",
                    "Entry Price": f"{entry_px:,.0f} TMN",
                    "Cost Basis": f"{cost:,.0f} TMN",
                    "Stop-Loss": f"{sl_px:,.0f} TMN",
                    "Take-Profit": f"{tp_px:,.0f} TMN",
                    "Peak Price": f"{peak_px:,.0f} TMN",
                    "Trailing Stop": trailing_status,
                })
            st.dataframe(pd.DataFrame(pos_table), width="stretch", hide_index=True)
        else:
            st.success("Zero open positions. 100% of balance is securely held in cash (TMN).")

    with ord_col:
        st.subheader("In-Flight & Resting Orders")
        pending_list = []
        if pending_buys:
            for sym, pb in pending_buys.items():
                age_s = int(time.time() - float(pb.get("created_at_ts", time.time())))
                pending_list.append({
                    "Side": "BUY",
                    "Symbol": sym,
                    "Price": f"{float(pb.get('limit_price', 0)):,.0f} TMN",
                    "Volume": f"{float(pb.get('order_tmn', 0)):,.0f} TMN",
                    "Age": f"{age_s}s / 180s",
                    "State": "RESTING LIMIT",
                })
        if pending_sells:
            for sym, ps in pending_sells.items():
                age_s = int(time.time() - float(ps.get("submitted_at_ts", time.time())))
                pending_list.append({
                    "Side": "SELL",
                    "Symbol": sym,
                    "Price": f"{float(ps.get('exec_price', 0)):,.0f} TMN",
                    "Volume": f"{float(ps.get('sell_qty', 0)):g} coins",
                    "Age": f"{age_s}s / 180s",
                    "State": "AWAITING FILL",
                })

        if pending_list:
            st.dataframe(pd.DataFrame(pending_list), width="stretch", hide_index=True)
        else:
            st.caption("No pending limit buy or resting sell orders. All previous orders fully settled.")


# ------------------------------------------------------------------
# TAB 3: AGENTIC AI STRATEGIST DECISION LOGS
# ------------------------------------------------------------------
with tab_ai:
    st.subheader("🧠 Immutable AI Decision Audit Stream")
    st.caption("Chronological records of LLM market evaluations, technical snapshots, and quantitative risk verdicts.")

    f1, f2, f3, f4 = st.columns([1, 1, 1, 1])
    sym_list = ["ALL"] + sorted(list(set(r.get("symbol", "") for r in logs if r.get("symbol"))))
    sel_sym = f1.selectbox("Filter Symbol", sym_list, index=0)
    sel_action = f2.selectbox("Filter Action", ["ALL", "BUY", "SELL", "HOLD"], index=0)
    sel_source = f3.selectbox("Filter Source", ["ALL", "LLM", "RISK_ENGINE"], index=0)
    max_display = f4.slider("Max Decisions", 5, 50, 20)

    # Filter logs
    view_logs = list(reversed(logs))
    if sel_sym != "ALL":
        view_logs = [r for r in view_logs if r.get("symbol") == sel_sym]
    if sel_action != "ALL":
        view_logs = [r for r in view_logs if r.get("action") == sel_action]
    if sel_source != "ALL":
        view_logs = [r for r in view_logs if r.get("decision_source") == sel_source]

    if not view_logs:
        st.warning("No decision logs match the selected filter criteria.")
    else:
        for entry in view_logs[:max_display]:
            act = entry.get("action", "HOLD")
            sym = entry.get("symbol", "")
            ts = entry.get("timestamp", "")[:19].replace("T", " ")
            conf = float(entry.get("ai_confidence", 0.0)) * 100.0
            verdict = entry.get("risk_verdict", "")
            snap = entry.get("market_snapshot", {})
            trade_meta = entry.get("trade_metrics") or {}
            source = entry.get("decision_source", "LLM")

            # Determine badge styling
            if act == "BUY":
                icon = "🟢 BUY"
            elif act == "SELL":
                icon = "🔴 SELL"
            else:
                icon = "⚪ HOLD"

            title_text = f"{icon} | {sym} | Conf: {conf:.0f}% | Source: {source} | Verdict: {verdict} | {ts}"
            with st.expander(title_text, expanded=(act in ("BUY", "SELL"))):
                # Render Persian AI Reasoning Card in RTL
                reasoning_fa = entry.get("reasoning_fa", "")
                if reasoning_fa:
                    st.markdown(
                        f"""<div class='rtl-box'>
                        <b>🧠 استدلال فنی هوش مصنوعی (Technical AI Reasoning):</b><br>
                        {reasoning_fa}
                        </div>""",
                        unsafe_allow_html=True,
                    )

                # Microstructure & Indicators Grid
                m_c1, m_c2, m_c3, m_c4 = st.columns(4)
                best_bid = float(snap.get("best_bid", 0))
                best_ask = float(snap.get("best_ask", 0))
                spread_val = float(snap.get("spread_pct", 0))
                m_c1.metric("Best Bid / Ask", f"{best_bid:,.0f}", f"Spread: {spread_val:.3f}%")

                rsi_val = float(snap.get("rsi_14", 0))
                is_bull = snap.get("trend_bullish", False)
                m_c2.metric("RSI(14) & Trend", f"{rsi_val:.1f}", "Bullish (EMA9>21)" if is_bull else "Bearish / Neutral")

                obi_val = float(snap.get("obi_top5", 0))
                taker_ratio = float(snap.get("taker_buy_ratio", 0.5)) * 100.0
                m_c3.metric("Order Book Imbalance (OBI)", f"{obi_val:+.3f}", f"Taker Buy: {taker_ratio:.1f}%")

                atr_pct = float(snap.get("atr_pct", 0))
                vol_z = float(snap.get("volume_zscore", 0))
                m_c4.metric("Volatility (ATR %)", f"{atr_pct:.2f}%", f"Vol Z-Score: {vol_z:+.2f}")

                # Observed Signal Chips
                signals = entry.get("signals_observed", [])
                if signals:
                    chips_html = " ".join([f"<span class='chip-badge chip-hold'>{s}</span>" for s in signals])
                    st.markdown(f"<b>Signals Observed:</b> {chips_html}", unsafe_allow_html=True)

                if trade_meta:
                    st.caption("Executed Trade Metrics:")
                    st.json(trade_meta)


# ------------------------------------------------------------------
# TAB 4: EVENT-DRIVEN BACKTEST RUNNER
# ------------------------------------------------------------------
with tab_backtest:
    st.subheader("⚡ Event-Driven Historical Backtest Engine")
    st.caption("Validates the strategy logic on historical Wallex OHLCV 15m candles with exchange fee modeling (0.25% Maker / 0.30% Taker).")

    bt_c1, bt_c2, bt_c3 = st.columns([2, 2, 1])
    target_sym = bt_c1.selectbox("Approved TMN Market", PRIORITY_ALLOWED_TMN_MARKETS, index=0)
    lookback = bt_c2.slider("Historical Lookback (Hours)", 48, 336, 168, step=24)

    if bt_c3.button("▶️ Run Backtest", type="primary", width="stretch"):
        with st.spinner(f"Fetching {lookback}h of 15m candles from Wallex API and simulating trades on {target_sym}..."):
            bt_cfg = AgentConfig(mode=ExecutionMode.BACKTEST)
            tester = StrategyBacktester(bt_cfg, symbol=target_sym, lookback_hours=lookback)
            report = tester.run()
            if report:
                st.success(f"Backtest Completed on {target_sym} ({report['candles_analyzed']} candles)!")
                r_c1, r_c2, r_c3, r_c4, r_c5 = st.columns(5)
                r_c1.metric("Final Capital", f"{report['final_capital_tmn']:,.0f} TMN", f"{report['return_pct']:+.2f}%")
                r_c2.metric("Net Realized PnL", f"{report['net_pnl_tmn']:+,.0f} TMN")
                r_c3.metric("Max Drawdown", f"-{report['max_drawdown_pct']:.2f}%", f"Limit: -{bt_cfg.max_portfolio_drawdown_pct*100:.1f}%")
                r_c4.metric("Total Trades & Win Rate", f"{report['total_trades']} Trades", f"Win Rate: {report['win_rate_pct']:.1f}%")
                r_c5.metric("Profit Factor", f"{report['profit_factor']:.2f}")
            else:
                st.error("Failed to fetch historical candles from Wallex. Check network connection.")


# ------------------------------------------------------------------
# TAB 5: RISK RULES & ARCHITECTURE
# ------------------------------------------------------------------
with tab_rules:
    st.subheader("🛡️ Institutional Risk Rules & Neuro-Symbolic Architecture")
    r_left, r_right = st.columns(2)

    with r_left:
        st.markdown(
            f"""
            #### 1. Dynamic Fractional Kelly Sizing
            - **High-Conviction Sizing (>= 80% Confidence + Bullish Trend + Positive OBI):** Up to **`{getattr(cfg, 'high_conviction_position_pct', 0.35)*100:.0f}%`** per trade.
            - **Moderate Sizing (65%-79% Confidence):** Standard **`{cfg.max_position_pct*100:.0f}%`** allocation.
            - **Liquid Cash Reserve Floor:** Minimum **`{cfg.min_cash_reserve_pct*100:.0f}%`** liquid TMN cash preserved at all times.
            - **Exchange Order Minimum Floor:** **`100,000 TMN`** strictly enforced post-rounding.

            #### 2. Sniper Mode Programmatic Hard Vetoes
            - **Trend Veto:** Rejects `BUY` if `trend_bullish is False` or `Price < EMA50`.
            - **Overbought Veto:** Rejects `BUY` if `bb_position > 0.75`.
            - **Depth Veto:** Rejects `BUY` if `obi_top5 < 0` (negative order book depth imbalance).
            - **Volatility Hurdle:** Rejects `BUY` if `ATR% < 0.35%` (flat ranges where exchange fees exceed potential movement).
            """
        )

    with r_right:
        st.markdown(
            f"""
            #### 3. Execution Protection & Liquidity Hole Guard
            - **Marketable Limit Orders:** Automatic conversion of market orders to LIMIT orders pegged to `best_ask` (buys) or `best_bid` (sells) to comply with Wallex exchange rules.
            - **Spread Guard:** Blocks stop-loss triggers if `spread_pct > 0.80%` and `mid_price > stop_loss_price`, preventing panic sells into temporary liquidity vacuums.
            - **Asynchronous Sell Queue:** Tracks in-flight sell orders in `pending_sell_orders` until 100% filled, eliminating false orphan balance alerts.

            #### 4. Pre-Deadline Liquidation
            - **Contest Auto-Flatten:** Automatically liquidates all active crypto inventory to Tomans (TMN) before deadline.
            - **Manual Flatten:** `python main.py --mode LIVE --flatten`
            """
        )

# ==========================================
# AUTO-REFRESH TIMER
# ==========================================
if auto_refresh:
    time.sleep(refresh_rate)
    st.rerun()
