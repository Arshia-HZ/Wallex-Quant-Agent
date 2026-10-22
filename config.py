# config.py
"""
Configuration module for Wallex AI Trading Agent.
Loads configuration from environment variables (.env) with production-ready defaults.
"""

import os
from enum import Enum
from dataclasses import dataclass, field
from typing import Tuple
from dotenv import load_dotenv

# Load variables from .env file if present
load_dotenv()


class ExecutionMode(str, Enum):
    BACKTEST = "BACKTEST"
    SHADOW = "SHADOW"   # Live market feed + AI reasoning, simulated order execution
    LIVE = "LIVE"       # Live execution on Wallex contest account


@dataclass
class AgentConfig:
    # Wallex Credentials & Endpoint
    api_key: str = field(default_factory=lambda: os.getenv("WALLEX_API_KEY", ""))
    base_url: str = field(default_factory=lambda: os.getenv("WALLEX_BASE_URL", "https://api.wallex.ir").rstrip("/"))

    # Execution Mode
    mode: ExecutionMode = field(
        default_factory=lambda: ExecutionMode(os.getenv("EXECUTION_MODE", "SHADOW").upper())
    )

    # Contest Capital & Allocation
    initial_capital_tmn: float = field(
        default_factory=lambda: float(os.getenv("INITIAL_CAPITAL_TMN", "5000000.0"))
    )
    allowed_symbols: Tuple[str, ...] = (
        "BTCTMN",
        "ETHTMN",
        "SOLTMN",
        "XRPTMN",
        "USDTTMN",
        "DOGETMN",
        "BNBTMN",
        "TRXTMN",
    )

    # Risk Management & Position Sizing
    max_position_pct: float = field(
        default_factory=lambda: float(os.getenv("MAX_POSITION_PCT", "0.22"))
    )  # Max 22% (1.1M TMN) per position
    min_cash_reserve_pct: float = field(
        default_factory=lambda: float(os.getenv("MIN_CASH_RESERVE_PCT", "0.20"))
    )  # Keep 20% (1.0M TMN) always in reserve
    max_concurrent_positions: int = field(
        default_factory=lambda: int(os.getenv("MAX_CONCURRENT_POSITIONS", "3"))
    )
    max_spread_pct: float = field(
        default_factory=lambda: float(os.getenv("MAX_SPREAD_PCT", "0.25"))
    )  # Skip illiquid order books with spread > 0.25%

    # Stop Loss & Take Profit
    hard_stop_loss_pct: float = field(
        default_factory=lambda: float(os.getenv("HARD_STOP_LOSS_PCT", "0.012"))
    )  # 1.2% hard stop-loss
    default_take_profit_pct: float = field(
        default_factory=lambda: float(os.getenv("DEFAULT_TAKE_PROFIT_PCT", "0.022"))
    )  # 2.2% target take-profit
    trailing_stop_activation_pct: float = field(
        default_factory=lambda: float(os.getenv("TRAILING_STOP_ACTIVATION_PCT", "0.010"))
    )  # Activate trailing stop when unrealized gain >= 1.0%
    trailing_stop_distance_pct: float = field(
        default_factory=lambda: float(os.getenv("TRAILING_STOP_DISTANCE_PCT", "0.007"))
    )  # Trail 0.7% behind peak price

    # Portfolio Drawdown Circuit Breaker
    max_portfolio_drawdown_pct: float = field(
        default_factory=lambda: float(os.getenv("MAX_PORTFOLIO_DRAWDOWN_PCT", "0.035"))
    )  # 3.5% total drawdown circuit breaker

    # Order Lifecycle & Stale Orders
    stale_order_timeout_seconds: int = field(
        default_factory=lambda: int(os.getenv("STALE_ORDER_TIMEOUT_SECONDS", "180"))
    )  # Cancel unfilled LIMIT orders after 3 minutes (180s)
    min_order_tmn: float = field(
        default_factory=lambda: float(os.getenv("MIN_ORDER_TMN", "150000.0"))
    )  # Safe buffer above Wallex 50,000 TMN minNotional
    fee_rate: float = field(
        default_factory=lambda: float(os.getenv("FEE_RATE", "0.002"))
    )  # 0.2% maker/taker fee for realistic shadow accounting

    # Loop Timing & Execution
    loop_interval_seconds: int = field(
        default_factory=lambda: int(os.getenv("LOOP_INTERVAL_SECONDS", "45"))
    )  # Scan frequency

    # Contest Liquidation Deadline
    contest_end_timestamp: int = field(
        default_factory=lambda: int(os.getenv("CONTEST_END_TIMESTAMP", "1792771200"))
    )
    flatten_minutes_before_end: int = field(
        default_factory=lambda: int(os.getenv("FLATTEN_MINUTES_BEFORE_END", "25"))
    )

    # Persistence & Audit Logging
    decision_log_file: str = field(
        default_factory=lambda: os.getenv("DECISION_LOG_FILE", "decision_logs.jsonl")
    )
    portfolio_state_file: str = field(
        default_factory=lambda: os.getenv("PORTFOLIO_STATE_FILE", "portfolio_state.json")
    )

    # LLM Settings (Configured for BoxAPI & GPT-6 Luna)
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
